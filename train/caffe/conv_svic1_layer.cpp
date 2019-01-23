#include <vector>

#include "caffe/layers/conv_svic1_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionSvic1Layer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim = (input_dim + 2 * pad_data[i] - kernel_extent)
        / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
void FixedPointQuantization(Dtype *data, int N, int bit_width, int frac_bits, float Gain = 1.0f) {

	Dtype max_data = (pow(2, bit_width - 1) - 1);
	Dtype min_data = -pow(2, bit_width - 1);

	for (int n = 0; n < N; n++) {

		data[n] *= Gain;
		data[n] /= pow(2, -frac_bits);

		data[n] = round(data[n]);
		data[n] = std::max(std::min(data[n], max_data), min_data);

		data[n] *= pow(2, -frac_bits);
	}
}

template <typename Dtype>
void coef_binary_quantization(Dtype *dst_ptr, const Dtype *src_ptr, const int Size, const int Mode)
{
	// 1: calculate var per filter:
	double var = 0;
	double max_abs = 0;
    for (int i = 0; i < Size; i++) {
		double abs_data = fabs(*(src_ptr + i));
		var += abs_data;
        if (max_abs < abs_data)
            max_abs = abs_data;
	}
	var /= Size;

	// 2: coef quantization: (0, +/-1, +/-2, +/-4) *var
	for (int i = 0; i < Size; i++) {

		Dtype coef = *(src_ptr + i);
		Dtype qcoef;

		if (Mode == QuantizationParameter_Precision_ONE_BIT) {
			// quantize to +/-1
			qcoef = (coef >= 0) ? var : -var;
		}
		else if (Mode == QuantizationParameter_Precision_TWO_BITS) {
			// quantize to 0,+/-1
			qcoef = (fabs(coef) < var/4) ? 0 : (coef >= 0) ? var : -var;
		}
		else if (Mode == QuantizationParameter_Precision_THREE_BITS) {
			// quantize to 0, +/-1, +/-2, +/-4
			Dtype abs_coef = fabs(coef);
			if (abs_coef > var * 3 / 4)
				qcoef = (coef >= 0) ? var : -var;
			else if (abs_coef > var / 2)
				qcoef = (coef >= 0) ? var / 2 : -var / 2;
			else if (abs_coef > var / 4)
				qcoef = (coef > 0) ? var / 4 : -var / 4;
			else
				qcoef = 0;
		}
		else if (Mode == QuantizationParameter_Precision_FIVE_BITS) {
			// quantize to -15 to +15 multiply by factor.
			double abs_qcoef = fabs(coef) * 15. / max_abs;
			qcoef = (coef >= 0) ? abs_qcoef : (-abs_qcoef);
			//qcoef = (fabs(coef) < var/4) ? 0 : (coef >= 0) ? var : -var;
		}
        else {
            printf("Error QuantizationParameter: Precision\n");
            exit(0);
        }

		*(dst_ptr + i) = qcoef;
	}

}


template <typename Dtype>
void ConvolutionSvic1Layer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {

	// layer input quantization:
	if (this->layer_param_.quantization_param().data_in_quantize() == true) {	//	if (this->phase_ == TEST) {
		int bit_width  = this->layer_param_.quantization_param().bw_data();
		int frac_bits  = this->layer_param_.quantization_param().fl_data();
		int shift_bits = this->layer_param_.quantization_param().data_in_shift_bits();
		float data_gain = pow(2, shift_bits);

		for (int i = 0; i < bottom.size(); i++) {
			Dtype *data = bottom[i]->mutable_cpu_data();
			int N = bottom[i]->count();
			FixedPointQuantization(data, N, bit_width, frac_bits, data_gain);
		}
	}

	//quantize coef and run forward process:
	int CoefPrecision = this->layer_param_.quantization_param().coef_precision();

    printf("svic foward cpu\n");

	if (CoefPrecision == QuantizationParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:
		// conv layer coef: no coef conversion.
		const Dtype* weight = this->blobs_[0]->cpu_data();

		// run convolution:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}
	else if ((CoefPrecision == QuantizationParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_TWO_BITS)||	  // two bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_THREE_BITS) || // three bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_FIVE_BITS)) {  // five bits coef scheme:
		// coef quantization: one bit
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		Dtype* src_weight = (Dtype*) this->blobs_[0]->cpu_data();
		Dtype* dst_weight = (Dtype*)blob_binary_coef.cpu_data();

		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);

		int offset = 0;
		for (int out_ch = 0; out_ch < Output_Channels; out_ch++) {
			for (int inp_ch = 0; inp_ch < Input_Channels; inp_ch++) {
				// process one filter:
				Dtype *src_ptr = src_weight + offset;
				Dtype *dst_ptr = dst_weight + offset;

				coef_binary_quantization(dst_ptr, src_ptr, Kernel_Height*Kernel_Width, CoefPrecision);

				offset += Kernel_Height * Kernel_Width;
			}
		}

		// run convolution:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, dst_weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}

	}	// end of one bit
	else if (CoefPrecision == QuantizationParameter_Precision_DYNAMIC_FIXED_POINT) {

		// coef quantization: Dynamic Fixed Point
		int bit_width = this->layer_param_.quantization_param().bw_params();
		int frac_bits = this->layer_param_.quantization_param().fl_params();
		float weight_gain = this->layer_param_.quantization_param().weight_gain();
		float bias_gain = this->layer_param_.quantization_param().bias_gain();

		Blob<Dtype> coef_blob0, coef_blob1;

		// blob[0]:
		coef_blob0.ReshapeLike(*this->blobs_[0]);
		caffe_copy(this->blobs_[0]->count(), (Dtype*)this->blobs_[0]->cpu_data(), (Dtype*)coef_blob0.cpu_data());
		FixedPointQuantization(coef_blob0.mutable_cpu_data(), coef_blob0.count(), bit_width, frac_bits, weight_gain);

		// blob[1]:
		if (this->bias_term_) {
			coef_blob1.ReshapeLike(*this->blobs_[1]);

			caffe_copy(this->blobs_[1]->count(), (Dtype*)this->blobs_[1]->cpu_data(), (Dtype*)coef_blob1.cpu_data());
			FixedPointQuantization(coef_blob1.mutable_cpu_data(), coef_blob1.count(), bit_width, frac_bits, bias_gain);
		}

		// run convolution:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->cpu_data();
			const Dtype* weight = coef_blob0.cpu_data();
			Dtype* top_data = top[i]->mutable_cpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_cpu_gemm(bottom_data + n * this->bottom_dim_, weight,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = coef_blob1.cpu_data();
					this->forward_cpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}  // end of fixed point
	else {
		printf("Error quantization scheme !!!\n");
	}

	// layer output quantization:
	if (this->layer_param_.quantization_param().data_out_quantize() == true) { // (this->phase_ == TEST) {
		int bit_width  = this->layer_param_.quantization_param().bw_data();
		int frac_bits  = this->layer_param_.quantization_param().fl_data();
		int shift_bits = this->layer_param_.quantization_param().data_out_shift_bits();
		float data_gain = pow(2, shift_bits);

		for (int i = 0; i < top.size(); i++) {
			Dtype *data = top[i]->mutable_cpu_data();
			int N = top[i]->count();

			FixedPointQuantization(data, N, bit_width, frac_bits, data_gain);
		}
	}

}

template <typename Dtype>
void ConvolutionSvic1Layer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	int CoefPrecision = this->layer_param_.quantization_param().coef_precision();
    printf("svic backward cpu\n");

	if (CoefPrecision == QuantizationParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:

		const Dtype* weight = this->blobs_[0]->cpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->cpu_diff();
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}
	}
	else if ((CoefPrecision == QuantizationParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_TWO_BITS)||	  // two bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_THREE_BITS) || // three bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_FIVE_BITS)) {  // five bits coef scheme:
		//std::cout << "======= Binary_Backward_cpu ============\n";

		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		const Dtype* src_weight = this->blobs_[0]->cpu_data();
		Dtype* dst_weight = blob_binary_coef.mutable_cpu_data();

		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);

		int offset = 0;
		for (int out_ch = 0; out_ch < Output_Channels; out_ch++) {
			for (int inp_ch = 0; inp_ch < Input_Channels; inp_ch++) {
				// process one filter:
				Dtype *src_ptr = (Dtype*)src_weight + offset;
				Dtype *dst_ptr = dst_weight + offset;
	
				coef_binary_quantization(dst_ptr, src_ptr, Kernel_Height*Kernel_Width, CoefPrecision);

				offset += Kernel_Height * Kernel_Width;
			}
		}

		// run backward prop:
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->cpu_diff();
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_cpu_gemm(top_diff + n * this->top_dim_, dst_weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}

	}
	else if (CoefPrecision == QuantizationParameter_Precision_DYNAMIC_FIXED_POINT) {

		// coef quantization: Dynamic Fixed Point
		int bit_width = this->layer_param_.quantization_param().bw_params();
		int frac_bits = this->layer_param_.quantization_param().fl_params();
		float weight_gain = this->layer_param_.quantization_param().weight_gain();
		float bias_gain = this->layer_param_.quantization_param().bias_gain();

		Blob<Dtype> coef_blob0, coef_blob1;

		// blob[0]:
		coef_blob0.ReshapeLike(*this->blobs_[0]);
		caffe_copy(this->blobs_[0]->count(), (Dtype*)this->blobs_[0]->cpu_data(), (Dtype*)coef_blob0.cpu_data());
		FixedPointQuantization(coef_blob0.mutable_cpu_data(), coef_blob0.count(), bit_width, frac_bits, weight_gain);

		/*
		// blob[1]: not updated or backprop ???
		if (this->bias_term_) {
			coef_blob1.ReshapeLike(*this->blobs_[1]);

			caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), coef_blob1.mutable_cpu_data());
			FixedPointQuantization(coef_blob1.mutable_cpu_data(), coef_blob1.count(), bit_width, frac_bits);
		}
		*/

		const Dtype* weight = coef_blob0.cpu_data();
		Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();

		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->cpu_diff();
			const Dtype* bottom_data = bottom[i]->cpu_data();
			Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_cpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_cpu_gemm(top_diff + n * this->top_dim_, weight,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}
	}
	else
	{
		printf("Error quantization scheme !!!\n");
	}
}

#ifdef CPU_ONLY
STUB_GPU(ConvolutionSvic1Layer);
#endif

INSTANTIATE_CLASS(ConvolutionSvic1Layer);
REGISTER_LAYER_CLASS(ConvolutionSvic1);
}  // namespace caffe
