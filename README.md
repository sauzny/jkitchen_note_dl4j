# jkitchen_note_dl4j

deeplearning4j 学习记录

教程列表 https://deeplearning4j.org/cn/tutorials

## 一、基本环境

- java 1.8
- maven 3+，配置好阿里云镜像
- CUDA，官网下载安装
- Spark，TODO
- Hadoop，TODO

## 二、准备dl4j-examples

clone demo `git@github.com:deeplearning4j/dl4j-examples.git`

然后在本机，执行 `mvn install`，下载所需依赖包

## 三、NN神经网络

- CNN(卷积神经网络)
- RNN(循环神经网络)
- DNN(深度神经网络)

### 3.1 CNN(卷积神经网络)

CNN 专门解决图像问题的，可用把它看作特征提取层，放在输入层上，最后用MLP 做分类。

### 3.2 RNN(循环神经网络)

RNN 专门解决时间序列问题的，用来提取时间序列信息，放在特征提取层（如CNN）之后。

### 3.3 DNN(深度神经网络)



## 四、应用

在语音识别中4层网络就能够被认为是“较深的”，而在图像识别中20层以上的网络屡见不鲜。
