#include <vector>

#include "caffe/layers/conv_svic1_layer.hpp"

namespace caffe {

template <typename Dtype>
void FixedPointQuantization(Dtype *data, int N, int bit_width, int frac_bits, float Gain = 1.0) {

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
__global__ void FixedPointQuantization_withShift_kernel(Dtype *data, const int N,
	const int bit_width, const int frac_bits, const int shift_bits) {

	Dtype max_data = (powf(2, bit_width - 1) - 1);
	Dtype min_data = -powf(2, bit_width - 1);
	Dtype gain1 = powf(2, shift_bits + frac_bits);
	Dtype gain2 = powf(2, -frac_bits);

	CUDA_KERNEL_LOOP(index, N) {
		Dtype val = gain1 * data[index];	// up-shift
		val = rintf(val);					// rounding
		val = (val < min_data) ? min_data : (val > max_data) ? max_data : val; // clamp
		data[index] = val * gain2;			// down-shift
	}
}

template <typename Dtype>
void coef_binary_quantization_(Dtype *dst_ptr, const Dtype *src_ptr, const int Size, const int Mode)
{
	// 1: calculate var per filter:
	double var = 0;
    double max_abs = 0; 
	for (int i = 0; i < Size; i++) {
        double abs_data = fabs(*(src_ptr+i));
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
			/*
            Dtype abs_coef = fabs(coef);
			if (abs_coef > var * 3 / 4)
				qcoef = (coef >= 0) ? var : -var;
			else if (abs_coef > var / 2)
				qcoef = (coef >= 0) ? var / 2 : -var / 2;
			else if (abs_coef > var / 4)
				qcoef = (coef > 0) ? var / 4 : -var / 4;
			else
				qcoef = 0;
            */
            //double QFactor = 4./(max_abs+0.0001);
            double QFactor = 4.f/(var+0.0001);
			int abs_coefInt = fabs(coef) * QFactor;
            abs_coefInt = (abs_coefInt < 3) ? abs_coefInt : 4;
            Dtype abs_coef = abs_coefInt / QFactor;
			qcoef = (coef >= 0) ? abs_coef : -abs_coef;
		}
		else if (Mode == QuantizationParameter_Precision_FIVE_BITS) {
			// quantize to -15 to +15 multiply by factor.
            double QFactor = 16./(max_abs+0.0001);
			int abs_coefInt = fabs(coef) * QFactor;
			Dtype abs_coef = abs_coefInt / QFactor;
			qcoef = (coef >= 0) ? abs_coef : -abs_coef;
		}
        else {
            printf("Error QuantizationParameter: Precision\n");
            exit(0);
        }
		*(dst_ptr + i) = qcoef;
	}
}

template <typename Dtype>
void ConvolutionSvic1Layer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

	// layer input quantization:
	if (this->layer_param_.quantization_param().data_in_quantize() == true) {	//	if (this->phase_ == TEST) {
		int bit_width  = this->layer_param_.quantization_param().bw_data();
		int frac_bits  = this->layer_param_.quantization_param().fl_data();
		int shift_bits = this->layer_param_.quantization_param().data_in_shift_bits();
		//float data_gain = pow(2, shift_bits);

		for (int i = 0; i < bottom.size(); i++) {
			Dtype *data = bottom[i]->mutable_gpu_data();
			const int N = bottom[i]->count();

			// NOLINT_NEXT_LINE(whitespace/operators)
			FixedPointQuantization_withShift_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
				data, N, bit_width, frac_bits, shift_bits);
			CUDA_POST_KERNEL_CHECK;
		}
	}
    
    //printf("svic foward gpu\n");

	//quantize coef and run forward process:
	int CoefPrecision = this->layer_param_.quantization_param().coef_precision();

	if (CoefPrecision == QuantizationParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_,
					this->blobs_[0]->gpu_data(),
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
        //printf("svic foward gpu: float \n");
	}
	else if ((CoefPrecision == QuantizationParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_TWO_BITS)||	  // two bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_THREE_BITS) || // three bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_FIVE_BITS)) {  // five bits coef scheme:
		// coef quantization: one or two bits
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		const Dtype* pt_origin_coef = this->blobs_[0]->cpu_data();	// in cpu
		Dtype* pt_binary_coef = blob_binary_coef.mutable_cpu_data();

		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);
        //printf("svic foward gpu: 1,2,3,5 bits -- %d \n", CoefPrecision);

		int offset = 0;
		for (int out_ch = 0; out_ch < Output_Channels; out_ch++) {
			for (int inp_ch = 0; inp_ch < Input_Channels; inp_ch++) {
				// process one filter:
                int coef_size = Kernel_Height * Kernel_Width;
				coef_binary_quantization_(pt_binary_coef+offset, pt_origin_coef+offset, coef_size, CoefPrecision);
				offset += coef_size;
			}
		}

		// run convolution:
        const Dtype* pt_coef = blob_binary_coef.gpu_data();
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, pt_coef,
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = this->blobs_[1]->gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}
	else if (CoefPrecision == QuantizationParameter_Precision_DYNAMIC_FIXED_POINT) {
		// printf("======= FixedPoint_Forward_gpu ============\n");
		// coef quantization: Dynamic Fixed Point
		int bit_width = this->layer_param_.quantization_param().bw_params();
		int frac_bits = this->layer_param_.quantization_param().fl_params();
		float weight_gain = this->layer_param_.quantization_param().weight_gain();
		float bias_gain = this->layer_param_.quantization_param().bias_gain();

		Blob<Dtype> coef_blob0, coef_blob1;

		// blob[0]:
		coef_blob0.ReshapeLike(*this->blobs_[0]);
		caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->cpu_data(), coef_blob0.mutable_cpu_data());
		FixedPointQuantization(coef_blob0.mutable_cpu_data(), coef_blob0.count(), bit_width, frac_bits, weight_gain);

		// blob[1]:
		if (this->bias_term_) {
			coef_blob1.ReshapeLike(*this->blobs_[1]);

			caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->cpu_data(), coef_blob1.mutable_cpu_data());
			FixedPointQuantization(coef_blob1.mutable_cpu_data(), coef_blob1.count(), bit_width, frac_bits, bias_gain);
		}

		// run convolution:
		for (int i = 0; i < bottom.size(); ++i) {
			const Dtype* bottom_data = bottom[i]->gpu_data();
			Dtype* top_data = top[i]->mutable_gpu_data();
			for (int n = 0; n < this->num_; ++n) {
				this->forward_gpu_gemm(bottom_data + n * this->bottom_dim_, coef_blob0.gpu_data(),
					top_data + n * this->top_dim_);
				if (this->bias_term_) {
					const Dtype* bias = coef_blob1.gpu_data();
					this->forward_gpu_bias(top_data + n * this->top_dim_, bias);
				}
			}
		}
	}
	else {
		printf("Error quantization scheme !!!\n");
	}

	// layer output quantization:
	if (this->layer_param_.quantization_param().data_out_quantize() == true) { // (this->phase_ == TEST) {
		int bit_width = this->layer_param_.quantization_param().bw_data();
		int frac_bits = this->layer_param_.quantization_param().fl_data();
		int shift_bits = this->layer_param_.quantization_param().data_out_shift_bits();
		//float data_gain = pow(2, shift_bits);

		for (int i = 0; i < top.size(); i++) {
			Dtype *data = top[i]->mutable_gpu_data();
			const int N = top[i]->count();

			// NOLINT_NEXT_LINE(whitespace/operators)
			FixedPointQuantization_withShift_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS >> >(
				data, N, bit_width, frac_bits, shift_bits);
			CUDA_POST_KERNEL_CHECK;
		}
	}

}

template <typename Dtype>
void ConvolutionSvic1Layer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

	//quantize coef and run forward process:
	int CoefPrecision = this->layer_param_.quantization_param().coef_precision();
    //printf("svic backward gpu\n");

	if (CoefPrecision == QuantizationParameter_Precision_FLOATING_POINT) {	// floating point coef scheme:

		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_gpu_gemm(top_diff + n * this->top_dim_, this->blobs_[0]->gpu_data(),
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}

	}
	else if ((CoefPrecision == QuantizationParameter_Precision_ONE_BIT) ||    // one bit coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_TWO_BITS)||	  // two bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_THREE_BITS)||  // three bits coef scheme:
	         (CoefPrecision == QuantizationParameter_Precision_FIVE_BITS)) {  // three bits coef scheme:
		//std::cout << "======= Binary_Backward_cpu ============\n";
		Blob<Dtype> blob_binary_coef;
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);

		const Dtype* pt_origin_coef = this->blobs_[0]->cpu_data();
		Dtype* pt_binary_coef = blob_binary_coef.mutable_cpu_data();

		// convert coefficeints:
		int Output_Channels = blob_binary_coef.shape(0);
		int Input_Channels = blob_binary_coef.shape(1);
		int Kernel_Height = blob_binary_coef.shape(2);
		int Kernel_Width = blob_binary_coef.shape(3);

		int offset = 0;
		for (int out_ch = 0; out_ch < Output_Channels; out_ch++) {
			for (int inp_ch = 0; inp_ch < Input_Channels; inp_ch++) {
		  		// process one filter:
                int coef_size = Kernel_Height * Kernel_Width;
				coef_binary_quantization_(pt_binary_coef+offset, pt_origin_coef+offset, coef_size, CoefPrecision);
				offset += coef_size;
            }
		}

		//std::cout << "===================\n";
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_gpu_gemm(top_diff + n * this->top_dim_,
							blob_binary_coef.mutable_gpu_data(),
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}
	}
	else if (CoefPrecision == QuantizationParameter_Precision_DYNAMIC_FIXED_POINT) {

		// printf("======= FixedPoint_Forward_gpu ============\n");
		// coef quantization: Dynamic Fixed Point
		int bit_width = this->layer_param_.quantization_param().bw_params();
		int frac_bits = this->layer_param_.quantization_param().fl_params();
		float weight_gain = this->layer_param_.quantization_param().weight_gain();
		float bias_gain = this->layer_param_.quantization_param().bias_gain();

		Blob<Dtype> blob_binary_coef, blob_binary_bias;

		// blob[0]:
		blob_binary_coef.ReshapeLike(*this->blobs_[0]);
        Dtype* pt_binary_coef = blob_binary_coef.mutable_gpu_data();
		
        caffe_copy(this->blobs_[0]->count(), this->blobs_[0]->gpu_data(), pt_binary_coef);
		FixedPointQuantization(pt_binary_coef, blob_binary_coef.count(), bit_width, frac_bits, weight_gain);

		// blob[1]:
		if (this->bias_term_) {
            blob_binary_bias.ReshapeLike(*this->blobs_[1]);
            Dtype* pt_binary_bias = blob_binary_bias.mutable_gpu_data();
			caffe_copy(this->blobs_[1]->count(), this->blobs_[1]->gpu_data(), pt_binary_bias);
			FixedPointQuantization(pt_binary_bias, blob_binary_bias.count(), bit_width, frac_bits, bias_gain);
		}


		//std::cout << "===================\n";
		Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
		for (int i = 0; i < top.size(); ++i) {
			const Dtype* top_diff = top[i]->gpu_diff();
			// Bias gradient, if necessary.
			if (this->bias_term_ && this->param_propagate_down_[1]) {
				Dtype* bias_diff = this->blobs_[1]->mutable_gpu_diff();
				for (int n = 0; n < this->num_; ++n) {
					this->backward_gpu_bias(bias_diff, top_diff + n * this->top_dim_);
				}
			}
			if (this->param_propagate_down_[0] || propagate_down[i]) {
				const Dtype* bottom_data = bottom[i]->gpu_data();
				Dtype* bottom_diff = bottom[i]->mutable_gpu_diff();
			    const Dtype* pt_coef = blob_binary_coef.gpu_data();
			    //const Dtype* pt_bias = blob_binary_bias.gpu_data();
				for (int n = 0; n < this->num_; ++n) {
					// gradient w.r.t. weight. Note that we will accumulate diffs.
					if (this->param_propagate_down_[0]) {
						this->weight_gpu_gemm(bottom_data + n * this->bottom_dim_,
							top_diff + n * this->top_dim_, weight_diff);
					}
					// gradient w.r.t. bottom data, if necessary.
					if (propagate_down[i]) {
						this->backward_gpu_gemm(top_diff + n * this->top_dim_, pt_coef,
							bottom_diff + n * this->bottom_dim_);
					}
				}
			}
		}
	}
	else {
		printf("Error quantization scheme !!!\n");
	}
}

INSTANTIATE_LAYER_GPU_FUNCS(ConvolutionSvic1Layer);

}  // namespace caffe
