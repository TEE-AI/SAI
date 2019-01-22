# TEE Simple AI (SAI) User Manual

## 简介
SAI是基于PyTorch的卷积神经网络模型训练，转换，部署工具——可训练得到定点化模型（1bit或者3bit），该模型可在TEE.COM的算力棒产品上运行。支持Windows，Linux等主流平台。

## 环境依赖

### 系统要求
目前支持在如下系统上运行
* Windows 10
* Ubuntu 16.04

### Pytorch版本
#### 0.4 or higher

#### 安装算力棒驱动
Windows 10系统会自动安装驱动，Linux系统需要一些额外的步骤，请参考以下命令为算力棒安装驱动

```
$ sudo cp api/lib/50-emmc.rules /etc/udev/rules.d/
```

## 模型训练(SAI_ROOT/train)

### 训练数据准备
将训练数据分为train和val两个目录，基于标签数目N，创建N个子目录，每个子目录中放入对应标签的图像数据。然后将train和val两个目录放置于SAI_ROOT/train/data目录下。

我们提供了一份猫狗图片分类数据集(百度网盘链接：https://pan.baidu.com/s/1la3C1d0xUBFhvkr0OJOl9w 提取码：ssjx)，该数据可用于训练一个两类(猫狗)分类器。同时我们还提供了一份在这个数据集上训练好的模型供参考。

| Model Name    | Top1 Acc(%) |
| --------- | -----:|
|[teeNet1](https://pan.baidu.com/s/1McEakAUyFqYjLKdUgnaj9w)       | 98.75
|[teeNet2](https://pan.baidu.com/s/1bXgtr3ksOGEH5F70dYBmNA)       | 97.25
|[teeNet3](https://pan.baidu.com/s/1DmaSE6xaOwoXm0cgnqH4NQ)       | 97.25

### 模型训练与转换
因为TEE算力棒仅支持VGG类型的卷积结构，所以SAI的模型训练工具也只提供了基于VGG类型的网络模型训练。当前版本支持三种类型的VGG网络：
- teeNet1: 标准的VGG16网络，包括13个卷积结构和3个全连接层
- teeNet2: 简化后的VGG网络，包括18个卷积结构，1个GAP层，1个全连接层
- teeNet3: 去掉全连接层后的VGG网络，包括16个卷积结构

SAI通过加载training.json文件来进行模型训练与转换，training.json文件放置在SAI_ROOT/train目录下，可通过文本编辑器对其进行编辑修改。training.json文件里的每个关键词描述如下：
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


## 推断部署(SAI_ROOT/api)

得到可以在算力棒上运行的模型后，我们可以通过调用SAI的API，在终端设备上很方便的部署上你的模型。

目前API里只包括用于图片分类任务的TEEClassifier library，支持windows/linux/arm-linux三个平台，后续会增加android/ios等平台支持。

在[SAI_ROOT/example](https://github.com/TEE-AI/SAI/tree/master/examples)下提供了各个平台的c++示例工程，展示了API的调用方法。

*Issue: Windows版本的library依赖于libeay32.dll，请确定系统路径里存在该dll文件。*


