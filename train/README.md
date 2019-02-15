### 基于Pytorch
要求pytorch version >= 0.4

#### 训练数据准备
将训练数据分为train和val两个目录，基于标签数目N，创建N个子目录，每个子目录中放入对应标签的图像数据。然后将train和val两个目录放置于SAI_ROOT/train/pytorch/data目录下。

我们提供了一份猫狗图片分类数
据集(百度网盘链接：https://pan.baidu.com/s/1la3C1d0xUBFhvkr0OJOl9w 提取码：ssjx)，该数据可用于训练一个两类(猫狗)分类器。同时我们还提供了一份在这个数据集上训练好的模型供参考。

| Model Name    | Top1 Acc(%) |
| --------- | -----:|
|[teeNet1](https://pan.baidu.com/s/1McEakAUyFqYjLKdUgnaj9w)       | 98.75
|[teeNet2](https://pan.baidu.com/s/1bXgtr3ksOGEH5F70dYBmNA)       | 97.25
|[teeNet3](https://pan.baidu.com/s/1DmaSE6xaOwoXm0cgnqH4NQ)       | 97.25

#### 模型训练与转换
因为TEE算力棒仅支持VGG类型的卷积结构，所以SAI的模型训练工具也只提供了基于VGG类型的网络模型训练。当前版本支持三种类型的VGG网络：
- teeNet1: 标准的VGG16网络，包括13个卷积结构和3个全连接层
- teeNet2: 简化后的VGG网络，包括18个卷积结构，1个GAP层，1个全连接层
- teeNet3: 去掉全连接层后的VGG网络，包括16个卷积结构

SAI通过加载training.json文件来进行模型训练与转换，training.json文件放置在SAI_ROOT/train/pytorch目录下，可通过文本编辑器对其进行编辑修改。training.json文件里的每个关键词描述如下：
* num_classes – 类别数目
* max_epoch – 最大迭代次数 
* learning_rate – 学习率 
* train_batch_size – 一次加载的训练数据数目 
* test_batch_size – 一次加载的测试数据数目 
* mask_bits – 每个主层的量化bit数
* act_bits – 每个主层的激活量化bit数
* resume – 接着之前中断的训练继续开始训练
* finetune – 加载一个预训练模型来微调
* full – 训练一个全精度的模型

在前面的工作都准备好后，在命令行窗口输入以下命令即可启动模型的训练与转换：

`$ python TEE_SAI.py`

运行结束后可在根目录下得到两个文件：conv.dat和fc.dat（如果是teeNet3网络，则只会得到conv.dat文件，因为该网络结构没有全连接层）。其中conv.dat是算力棒上加载运行的模型。

*Tips: 关于模型训练，我们建议先使用full模式训练一个全精度的模型F，再通过加载这个全精度模型F来finetune训练量化模型，得到最终的可部署模型。*

### 基于Caffe
#### 模型训练
假定你已经有自己的Caffe环境，那么参考以下几步就可以让你的caffe可以训练量化模型了：
* Step 1: 将本目录下的caffe.proto文件与CAFFE_DIRECTORY/src/caffe/proto merge一下，注意：不是覆盖
* Step 2: 将本目录下的conv_svic1_layer.cpp和conv_svic1_layer.cu拷贝到CAFFE_DIRECTORY/src/caffe/layer
* Step 3：将本目录下的conv_svic1_layer.hpp拷贝到CAFFE_DIRECTORY/include/caffe/layer
* Step 4: 重新编译caffe

将卷积量化层加入到Caffe后，训练量化模型就很简单了，只需要在train.prototxt文件里用ConvolutionSvic1替换Convolution即可。

```
layer {
  bottom: "data"
  top: "conv1_1"
  name: "conv1_1"
  type: "ConvolutionSvic1"  #use our quantization layer
  param {
    lr_mult: 1
    decay_mult: 1
  }
  param {
    lr_mult: 2
    decay_mult: 0
  }
  convolution_param {
    num_output: 64
    pad: 1
    kernel_size: 3
    weight_filler {
      type: "gaussian"
      std: 0.01
    }
    bias_filler {
      type: "constant"
      value: 0
    }
	quantization_param {  #define the quantization param
      coef_precision: 4  #means 3 bit
    }
  }
}
```

具体示例可以参考teeNet1_quantization目录下的train.prototxt，它定义的是一个标准VGG16网络。

#### 模型转换

训练得到带量化的模型后，使用convert_tool下的脚本convt_cnnsvic.py即可完成将带量化模型转换为TEE算力棒上可以运行的模型了。

```
python convt_cnnsvic.py --weights teeNet1.caffemodel --network teeNet1.prototxt --config teeNet1.json  --testdata test.bmp --output .
```


