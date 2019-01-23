# TEE Simple AI (SAI) User Manual

SAI是基于PyTorch的卷积神经网络模型训练，转换，部署工具——可训练得到定点化模型（1bit或者3bit），该模型可在TEE.COM的算力棒产品上运行。支持Windows，Linux等主流平台。

## 环境依赖

### 系统 (Win10/Ubuntu 16.04)

### 安装算力棒驱动
Windows 10系统无需安装驱动，Linux系统请参考以下命令为算力棒安装驱动

```
$ sudo cp api/lib/50-emmc.rules /etc/udev/rules.d/
```

## 模型训练([SAI_ROOT/train](https://github.com/TEE-AI/SAI/tree/master/train))
针对不同的开发者，我们提供了两种训练框架(Pytorch与Caffe)的模型训练与转换，大家可以根据个人喜好选择使用。

## 推断部署(SAI_ROOT/api)

得到可以在算力棒上运行的模型后，我们可以通过调用SAI的API，在终端设备上很方便的部署上你的模型。

目前API里只包括用于图片分类任务的TEEClassifier library，支持windows/linux/arm-linux三个平台，后续会增加android/ios等平台支持。

在[SAI_ROOT/example](https://github.com/TEE-AI/SAI/tree/master/examples)下提供了各个平台的c++示例工程，展示了API的调用方法。

*Issue: Windows版本的library依赖于libeay32.dll，请确定系统路径里存在该dll文件。*

## 交流群
QQ: 781349981


