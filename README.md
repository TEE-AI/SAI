# TEE Simple AI (SAI) 用户手册V1.0
本文档内容来自于北京梯易易科技有限公司（TEE），基于本文档内容，可用于评估本公司产品的性能，本文档包括环境创建，基于Pytorch工具的量化模型训练，模型转换，以及如何快速部署在Windows, Linux等平台。

## SAI概要
SAI是基于PyTorch的卷积神经网络模型训练，转换，部署工具——可用于将float型卷积神经网络模型转换为定点量化模型（1bit或者3bit），并可通过TEE公司的算力棒来运行，或者从头开始训练量化模型。部署的时候可使用主机的CPU和TEE公司的算力棒通过联合通信进行推断，支持Windows，Linux等主流平台。

基于SAI和本公司出品的算力棒，可以非常方便的训练一个精度损失较低的量化模型，并转换成可以在算力棒上运行的模型，基于转换好的模型，最后SAI还为开发者提供了快速的部署到Windows, Linux等平台的一键部署工具。

## 硬件与系统要求
SAI运行环境对主机配置的相关要求如下：
* CPU >= Intel i5 (推荐i7) 
* 内存 >=8 GB

当前支持在如下系统上运行
* Windows 10
* Ubuntu LTS 16.04

## 软件环境依赖
### Python
推荐直接安装anaconda集成python环境，python2.7或者python3.7均可，可在 https://www.anaconda.com/download/ 上根据自己的系统选择下载Windows或者Linux的安装包进行安装。

安装完成后可以在控制台（Windows下打开Windows Command Prompt， Linux下打开Terminal）输入以下命令来确认python环境是否安装成功

```
$ python
Python 3.7.0 (default, Jun 28 2018, 08:04:48) [MSC v.1912 64 bit (AMD64)] :: Anaconda, Inc. on win32
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### Pytorch
Pytorch是Facebook开源的一款神经网络框架，可在官网 https://pytorch.org/ 自行选择和你的环境相符合的下载命令进行安装，安装完成后，可在控制台（Windows下打开Windows Command Prompt， Linux下打开Terminal）输入以下命令来确认pytorch环境是否安装成功

```
>>> import torch
>>> import torchvision
```

### CUDA (可选)
推荐使用GPU来跑训练，CPU跑训练实在是太慢，你会无法忍受的，^o^.     

若要使用GPU来训练，就需要安装CUDA.一般来说，需要通过以下几步安装CUDA
1.安装NVIDIA显卡驱动，去官网 (http://www.nvidia.cn/Download/index.aspx?lang=cn) 查找适配自己电脑GPU的驱动
2.安装CUDA9.0，去官网 (https://developer.nvidia.com/cuda-90-download-archive) 下载
3.安装cuDNN，去官网 (https://developer.nvidia.com/cudnn) 下载

可通过以下命令来确认GPU是否成功安装:

```
>>> import torch
>>> torch.cuda.is_available()
```


## 安装算力棒驱动
将SAI_v1.0.zip解压，Windows 10系统会自动安装驱动，Linux系统需要一些额外的步骤，请参考以下命令为算力棒安装驱动

```
$ sudo cp lib/libftd3xx.so.0.5.21 /usr/lib/ 
$ sudo cp lib/*.rules /etc/udev/rules.d/
```

## 训练数据准备
将你的训练数据分为train和val两个目录，基于标签数目N，创建0 -- N-1个子目录，每个子目录中放入对应标签的图像数据。然后将train和val两个目录放置于SAI_ROOT/data目录下。

## 模型训练与转换
因为TEE算力棒仅支持VGG类型的卷积结构，所以SAI的模型训练工具也只提供了基于VGG类型的网络模型训练。当前版本支持三种类型的VGG网络：
- teeNet1: 标准的VGG16网络，包括13个卷积结构和3个全连接层
- teeNet2: 简化后的VGG网络，包括18个卷积结构，1个GAP层，1个全连接层
- teeNet3: 去掉全连接层后的VGG网络，包括16个卷积结构

SAI通过加载training.json文件来进行模型训练与转换，training.json文件放置在SAI_ROOT目录下，可通过文本编辑器对其进行编辑修改。training.json文件里的每个关键词描述如下：
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

在前面的工作都准备好后，你可以在命令行窗口输入以下命令来启动模型的训练与转换工作：

`$ python TEE_SAI.py`

运行结束后可在根目录下得到两个文件：conv.dat和fc.dat（如果是teeNet3网络，则只会得到conv.dat文件，因为该网络结构没有全连接层）。其中conv.dat是算力棒上加载运行的模型。

*Tips: 关于模型训练，我们建议先使用full模式训练一个全精度的模型F，再通过加载这个全精度模型F来finetune训练量化模型，得到最终的可部署模型。*

## 推断部署
通过前面的模型训练与转换步骤，得到了可以在算力棒上运行部署的模型，接下来我们可以通过SAI的infer工具，结合分类任务，将该模型快速的部署到终端设备上。TEE_SAI SDK目前支持windows/linux/arm-linux三个平台的推断部署，后续会增加android/ios等平台支持。

<font color=#FF0000>当前版本仅提供了针对teeNet1网络结构的分类任务推断部署。</font>

下面我们详细介绍3种平台的推断的编译和部署。首先进入SAI_ROOT/infer/目录，可根据实际需要部署的平台选择Windows，Linux或者Arm-Linux文件夹下的部署工具。

| 平台 | 依赖 | 描述 |
| ------ | ------ | ------ |
| Windows | OpenCV/OpenBLAS/FFmpeg | TEE发布包中已包含，无需编译 |
| Linux | OpenCV/FFmpeg | TEE发布包中已包含，需要时编译 |
| ARM Linux | OpenCV/FFmpeg/QT | TEE发布包中已包含，需要时编译 |

此处FFmpeg和QT依赖只是用于显示demo和界面。实际部署时可以根据使用场景选择是否去掉。

*Tips：以Windows平台为例，请将前面转换好的conv.dat和fc.dat文件拷贝到SAI_ROOT/infer/windows/bin/model目录下，运行run.bat即可看到演示界面。如果需要修改类别数或者输出显示方式或者其他后处理，可以打开SAI_ROOT/infer/windows/目录下的TEE_SAI.sln工程自行修改定制。*

### Windows平台

| 文件 | 功能 | 描述 |
| ------ | ------ | ------ |
| windows/TEE_SAI.sh | windows平台推断工程 | 需要Visual Studio 2015版本及以上 |
| windows/lib | windows平台编译和运行需要的静态库和动态库 |  |
| windows/bin | windows平台编译输出和运行目录 | 将model文件夹复制到此目录下，直接双击运行run.bat |

### Linux平台

| 文件 | 功能 | 描述 |
| ------ | ------ | ------ |
| linux/CMakeLists.txt | Linux平台编译文件 |  |
| linux/lib | Linux平台编译和运行需要的静态库和动态库 |  |
| linux/build | Linux平台独立编译目录 | `cd build`<br>` cmake .`<br>`make` |
| linux/bin | Linux平台运行目录 | 将编译生成的可执行文件TEEClassifierDemo和model文件夹拷贝到此目录，运行run.sh |

### ARM Linux平台

| 文件 | 功能 | 描述 |
| ------ | ------ | ------ |
| arm64/CMakeLists.txt | ARM64 Linux平台编译文件 |  |
| arm64/lib | ARM64 Linux平台编译和运行需要的静态库和动态库 |  |
| arm64/build | ARM64 Linux平台独立编译目录 | `cd build`<br>` cmake-gui .`<br>`make` |
| arm64/bin | ARM64 Linux平台运行目录 | 将编译生成的可执行文件TEEClassifierDemo和model文件夹拷贝到此目录，运行run.sh |

编译aarch64 linux时需要使用linaro的交叉编译工具，SAI_ROOT/arm64/toolchains/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu.tar.xz为交叉编译工具。



