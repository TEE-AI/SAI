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
将你的训练数据分为train和val两个目录，基于标签数目N，创建0~N-1个子目录，每个子目录中放入对应标签的图像数据。然后将train和val两个目录放置于SAI_ROOT/train/data目录下。

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

在前面的工作都准备好后，你可以在命令行窗口输入以下命令来启动模型的训练与转换工作：

`$ python TEE_SAI.py`

运行结束后可在根目录下得到两个文件：conv.dat和fc.dat（如果是teeNet3网络，则只会得到conv.dat文件，因为该网络结构没有全连接层）。其中conv.dat是算力棒上加载运行的模型。

*Tips: 关于模型训练，我们建议先使用full模式训练一个全精度的模型F，再通过加载这个全精度模型F来finetune训练量化模型，得到最终的可部署模型。*


## 推断部署(SAI_ROOT/api)

通过前面的模型训练与转换步骤，得到了可以在算力棒上运行部署的模型，接下来我们可以通过SAI的Infer工具(TEE-AI SDK)，结合分类任务，将该模型快速的部署到终端设备上。

目前Infer工具只包括用于图片分类的TEEClassifier SDK，支持windows/linux/arm-linux三个平台的推断部署，后续会增加android/ios等平台支持。

### TEE-AI SDK
所有的推断和部署需要供助于我们所发布的工具库(位于SAI_ROOT/api目录下)来实施，工具库提供了创建推荐引擎，传入样本进行推断，以及资源清理等步骤的函数接口，下面结合头文件INXInferenceEngine.h详细介绍该工具库的使用流程。

#### 创建引擎
创建推断引擎之前需要先设定好引擎参数，具体的数据结构如下：

```
/* Inference engine send result data callback function */
typedef NXRet(*LPResultCB)(nxvoid *pPrivateData, nxui8 const *retBuf, nxi32 bufLen, nxui64 id, nxi32 classNum);

/* Inference engine config */
typedef struct {
	nxi32 stickNum;				// 使用算力棒的个数
	nxi32 threadNum;			// 引擎启动的线程数
	nxi32 netType;				// 网络类型：1-teeNet1, 2-teeNet2, 3-teeNet3
	nxi32 classNum;				// 分类结果的类数
	nxi8 const *modelPath; 		// 算力棒模型/后处理模型文件所在目录(绝对路径)
	nxi8 const *stickCNNName; 	// 算力棒模型文件名(绝对路径)
	nxi8 const *hostNetName; 	// 后处理模型文件名(绝对路径)
	LPResultCB pCB;				// 引擎回调函数
	nxvoid *pCBData;			// 引擎回调函数自定义参数(即pPrivateData)
} NXEngineConf;
```

设定好引擎参数后，同时传入一个`nxvoid**`类型的变量，调用以下函数生成引擎，生成的引擎由`*ppEngine`指向的内存区域所表示。回调函数将在下一小节介绍。

```
/*  Create inference engine */
NXRet NXDLL NXCreateInferenceEngine(nxvoid **ppEngine, NXEngineConf const *pConf);
```

#### 调用引擎进行推断
调用引擎前需要按照指定格式准备好图像数据，其对应的数据结构为

```
// 像素格式
typedef enum {
	ePixfmtBGR = 0, // BGRBGR
	ePixfmtRGB = 1, // RGBRGB
} NXPixFmt;

// 图像
typedef struct {
	nxi32 w;			// 图像宽度
	nxi32 h;			// 图像高度
	NXPixFmt pixfmt;	// 像素格式
	nxui8 *data;		// 具体数据
} NXImg;
```

准备完成后，调用以下接口对图像进行推断，其中engine参数即为上一步中所创建的引擎。该接口以非阻塞模式执行，函数将直接返回并在推断在完成后执行`NXEngineConf.pCB`回调函数，并以`NXEngineConf.pCBData`作为回调函数的第一个参数传入。回调函数可以由用户自行定制。

```
/* send an image to engine and engine set *pID value. engine will send the id to callback function */
NXRet NXDLL NXPushTask(nxvoid *engine, NXImg const *pImg, nxui64 *pID);
```

#### 清理收尾工作并销毁引擎
以下函数将等待所有已经被Push的任务处理完成

```
/* clear all task */
NXRet NXDLL NXClearAllTask(nxvoid *engine);
```

确认不再使用引擎时可调用以下接口销毁引擎

```
/* destroy inference engine */
NXRet NXDLL NXDestroyInferenceEngine(nxvoid *pEngine);
```

### 使用样例(DEMO)
位于SAI_ROOT/examples目录下的代码中提供了用于图片分类的TEEClassifier SDK的使用样例，详细内容请见该目录下的README文件。


