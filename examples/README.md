# TEE_AI SDK使用示例文档

本文档将描述使用TEE_AI SDK进行推断部署的实际样例，我们首先搭建了一套可复用的代码框架，开发者可以选择性地复用或扩充此框架，以适应其它形式的应用部署。在此框架之上，我们开发了一个用于图片鉴黄的实际样例，供开发者参考和使用。

目前我们只提供了C++语言开发的DEMO样例，Python版本将于日后提供。

## 测试平台
以下代码在windows 10 + opencv3.4.3 + vs2015平台下测试通过

## 代码框架
代码框架分为`Reader`, `Preprocessor`, `EngineWrapper`, `Launcher`四个主要组件
* Reader
  * 负责从文件列表中依次读取样本，每次调用`get`方法返回一个样本
  * 可通过注册引入一个`Preprocessor`实例，此时`get`方法会在返回处理后的图像
  * 目前只实现了`ImageReader`子类
* Preproccessor
  * 基类只定义了接口，传入一个`cv::Mat`类型的图像，返回处理后的图像
  * 用户可根据需求生成子类并进行定制
* EngineWrapper
  * SDK接口函数的简单封装，创建时需要传入`EngineConfig*`类型参数
  * `EngineConfig`中需指定模型路径，Engine所启用的线程数等参数，同时需要指定Engine处理每个样本后的回调函数及参数
  * 对外暴露`create`, `push`, `clear`方法，并在析构时清理所占资源
* Lanucher
  * 负责组装以上三个组件并启动执行流程
  * 创建时接收`Reader*`, `Preprocessor*`, `EngineWrapper*`三个类型的参数
* 调用`run()`方法启动执行

## DEMO开发步骤
开发者可以使用以上框架进行DEMO开发，具体步骤为
1. 根据需要定制`Preprocessor`: 继承`Preprocessor`基类，实现`process`方法，该方法接收一个`cv::Mat`做为输入，返回处理后的`cv::Mat`
2. 根据需要定制回调函数及函数参数：回调函数类型请见`LPResultCB`
3. 编写程序，主要步骤为
    3.1 调用`ParseArgs`解析命令行参数
    3.2 创建`ImageReader`及第1步中自定义的`Preprocessor`子类的实例
    3.3 创建`EngineWrapper`实例
        3.3.1 生成`NXEngineConfig`: 调用`GenerateEngineConfigFromCmdArgs`从命令行参数生成`NXEngineConfig`的基础字段，再将第2步中定义的回调函数及参数赋给`NXEngineConfig`的`pCB`, `pCBData`字段
        3.3.2 将`NXEngineConfig`传入`EngineWrapper`构造函数，生成`EngineWrapper`实例
    3.4 创建`Launcher`实例，将`Reader`, `Preprocessor`和`EngineWrapper`实例传入其构造函数
    3.5 调用`Launcher`的`run`方法，启动执行流程
    3.6 清理资源并销毁引擎
4. 代码位置: cpp/src/demo.cpp




