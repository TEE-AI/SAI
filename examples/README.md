# TEEClassifier SDK使用示例文档

本文档将描述使用TEEClassifier SDK进行推断部署的实际样例，我们首先搭建了一套可复用的代码框架，开发者可以选择性地复用或扩充此框架，以适应其它形式的应用部署。在此框架之上，我们开发了一个用于图片分类的实际样例，供开发者参考和使用。

目前我们只提供了C++语言开发的DEMO样例。

## 测试平台
以下代码在Windows 10 + Visual Studio 2015 / Ubuntu 16.04平台下测试通过

## 代码框架
代码框架分为`Reader`, `Preprocessor`, `EngineWrapper`, `Launcher`四个主要组件
* Reader
  * 负责从文件列表中依次读取样本，每次调用`get`方法返回一个样本
  * 可通过注册引入一个`Preprocessor`实例，此时`get`方法会在返回处理后的图像
  * 目前只实现了`ImageReader`子类
* Preproccessor
  * 图像预处理器，需要与训练模型之前的预处理流程保持一致
  * 基类只定义了接口，传入一个`cv::Mat`类型的图像，返回处理后的图像
  * 用户可根据需求生成子类并进行定制
* EngineWrapper
  * SDK接口函数的简单封装，创建时需要传入`NXEngineConfig*`类型参数
  * `NXEngineConfig`中需指定模型路径，Engine所启用的线程数等参数，同时需要指定Engine处理每个样本后的回调函数及参数
  * 对外暴露`create`, `push`, `clear`方法，并在析构时清理所占资源
* Lanucher
  * 负责组装以上三个组件并启动执行流程
  * 创建时接收`Reader*`, `Preprocessor*`, `EngineWrapper*`三个类型的参数
  * 调用`run()`方法启动执行

## DEMO开发步骤
开发者可以使用以上框架进行DEMO开发，具体步骤为
* 根据需要定制`Preprocessor`: 继承`Preprocessor`基类，实现`process`方法，该方法接收一个`cv::Mat`做为输入，返回处理后的`cv::Mat`
* 根据需要定制回调函数及函数参数：回调函数类型请见`LPResultCB`
* 编写程序，主要步骤为
    * 调用`ParseArgs`解析命令行参数
    * 创建`ImageReader`及第1步中自定义的`Preprocessor`子类的实例
    * 创建`EngineWrapper`实例
        * 生成`TEEClsConf`: 调用`readFileIntoString`从命令行参数输入的'configFile'字段的json文件中解析出引擎参数，再将第2步中定义的回调函数及参数赋给`NXEngineConfig`的`pCB`, `pCBData`字段
        * 将`TEEClsConf`传入`EngineWrapper`构造函数，生成`EngineWrapper`实例
    * 创建`Launcher`实例，将`Reader`, `Preprocessor`和`EngineWrapper`实例传入其构造函数
    * 调用`Launcher`的`run`方法，启动执行流程
    * 清理资源并销毁引擎

## 样例-图片分类
我们开发了一个用于图片分类的样例，开发者可参考使用。

### 目录结构
```
src
   |--- demo.cpp, *.cpp, *.h
windows
   |--- TEE_SAI.sln
   `--- bin
         |--- TEEClassifierDemo.exe
         |--- TEEClassifier.dll, *.dll(from 3rdparty)
         `--- model
              |--- data
              |--- classification.json
              |--- image.list
              |--- label.txt
linux
   |--- CMakeLists.txt

```

### Windows执行步骤
* 下载第三方库，将其置于SAI_ROOT/examples/3rdparty目录中
  * 百度网盘下载：https://pan.baidu.com/s/1O3IxeB1RRokXwphMoQBJug 提取码：fcl0 
* 编译工程在cpp/windows/bin中生成TEEClassifierDemo.exe
* 将SAI_ROOT/api/lib/TEEClassifier.dll中的文件复制到cpp/windows/bin之中
* 创建cpp/windows/bin/model目录，其中data为存放样本的目录，classification.json为调用引擎所需的参数，image.list为对应的文件列表(相对路径)，label.txt为分类类别对应的文字名称，classification.json的详细说明为：
******
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
****
* 填入TEEClassifierDemo.exe所需要的参数，执行即可。
* 如下命令可做为参考：

```
TEEClassifierDemo.exe configFile "classification.json" labelName "label.txt" fileLise "image.list"

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

* 运行相关设置可参考Windows上的设置，将运行时依赖的opencv等依赖库拷贝到和TEEClassifierDemo一起，创建model目录，拷贝模型文件等。然后参考以下命令即可看到实际运行效果：

```
./TEEClassifierDemo configFile "classification.json" labelName "label.txt" fileLise "image.list"

```

### TEEClassifierDemo参数说明

```
-------- Inference Engine ---------
*   configFile  c_string
*   dataTestPath   c_string(default path is ./model)
*   labelName   c_string
*   fileList    c_string
-----------------------------------
```


