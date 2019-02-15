# TEESAI SDK使用示例文档

# TEEClassifier API
本文档将描述使用TEEClassifier API进行推断部署的实际样例，我们搭建了一套可复用的代码框架，开发者可以选择性地复用或扩充此框架，以适应其它形式的应用部署。在此框架之上，我们开发了一个用于图片分类的实际样例，供开发者参考和使用。

该代码在Windows 10 + Visual Studio 2015 / Ubuntu 16.04平台下测试通过

### Windows执行步骤
* 下载第三方库，将其置于SAI_ROOT/examples/3rdparty目录中
  * 百度网盘下载：https://pan.baidu.com/s/1O3IxeB1RRokXwphMoQBJug 提取码：fcl0 
* 编译工程在cpp/windows/bin/Classification/中生成TEEClassifierDemo.exe
* 将SAI_ROOT/api/lib/TEEClassifier.dll中的文件复制到cpp/windows/bin/Classification之中
* 创建cpp/windows/bin/Classification/model目录，其中data为存放样本的目录，classification.json为调用引擎所需的参数，image.list为对应的文件列表(相对路径)，label.txt为分类类别对应的文字名称，classification.json的详细说明为：

```
{
  "Root Path":"D:/SAI/model/", #"Set the path to your model file, make sure conv file and fc file are under the same path"
  "Stick Begin ID": 0, #The first available eMMC device node. This node is newly created after eMMC dongle is plugged in
  "Stick Delay Time": 7000, # Delay time (us) between eMMC USB dongle write and read commands, 5000 for gNet3, 12000 for gNet1
  "Stick Num": 1, #stick number
  "Thread Num": 6, #thread number
  "Classify Conf":{  
    "Class Num": 2, # the class number you want to classify
    "Net Type": 2, #net type 
    "Conv File": "conv.dat", # convolution parameters used in stick
    "FC File":"fc.dat"   # full connetion parameters used in stick
  }
}
```

* 填入TEEClassifierDemo.exe所需要的参数，执行即可。
* 如下命令可做为参考：

```
TEEClassifierDemo.exe configFile "classification.json" labelName "label.txt" fileList "image.list"

```

### Linux执行步骤

* 打开CMakeLists.txt, 定位到第47行，设置opencv的路径

```
#Set the path to your opencv directory
set(DEP_PATH /path to your opencv directory)
```

* 参考以下命令进行编译,编译完成可得到可执行文件：TEEClassifierDemo

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

* 运行相关设置可参考Windows上的设置

# TEEDetectorFRCNN API
以下将描述使用TEEDetectorFRCNN API进行推断部署的实际样例，我们搭建了一套可复用的代码框架，开发者可以选择性地复用或扩充此框架，以适应其它形式的应用部署。在此框架之上，我们开发了一个用于安全帽检测的实际样例，供开发者参考和使用。

该代码在Windows 10 + Visual Studio 2015 / Ubuntu 16.04平台下测试通过

### Windows执行步骤
* 下载第三方库，将其置于SAI_ROOT/examples/3rdparty目录中
  * 百度网盘下载：https://pan.baidu.com/s/1O3IxeB1RRokXwphMoQBJug 提取码：fcl0 
* 编译工程在cpp/windows/bin/Detection_FasterRcnn/中生成TEE_Detection_FasterRcnn.exe
* 将SAI_ROOT/api/lib/TEEDetectorFRCNN.dll中的文件复制到cpp/windows/bin/Detection_FasterRcnn之中
* 创建cpp/windows/bin/Detection_FasterRcnn/model目录，其中data为存放样本的目录，detection.json为调用引擎所需的参数，image.list为对应的文件列表(相对路径),detection.json的详细说明为：

```
{
  "Root Path": D:/SAI/model/", #"Set the path to your model file, make sure conv file and fc file are under the same path"
  "Stick Begin ID": 0, #The first available eMMC device node. This node is newly created after eMMC dongle is plugged in
  "Stick Delay Time": 12000,# Delay time (us) between eMMC USB dongle write and read commands, 5000 for gNet3, 12000 for gNet1
  "Stick Num": 1, #stick number
  "Thread Num": 6,#thread number
  "FRCNN Conf":{  
    "Conv File": "vgg16.dat",# convolution parameters used in stick
    "Proto File":"rpn_rcnn.prototxt",# caffe prototxt file run in PC 
    "Weights File":"rpn_rcnn.caffemodel",# caffe caffemodel run in PC
	"Neg Threshold":0.03 #negative threshold that used for detection filter
  }
}
```

* 填入TEE_Detection_FasterRcnn.exe所需要的参数，执行即可。
* 如下命令可做为参考：

```
TEE_Detection_FasterRcnn.exe configFile "detection.json"  fileList "image.list"

```

### Linux执行步骤

* 打开CMakeLists.txt, 定位到第47行，设置opencv的路径

```
#Set the path to your opencv directory
set(DEP_PATH /path to your opencv directory)
```

* 参考以下命令进行编译,编译完成可得到可执行文件：TEE_Detection_FasterRcnn

```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

* 运行相关设置可参考Windows上的设置




