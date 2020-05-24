# Task02：数据读取与数据扩增
## 1 Baseline导入包学习
### 1.1 pandas,numpy,os, sys, glob, shutil, json
常见的文件导入，读取，数据处理，科学计算等
### 1.2 cv2,PIL
cv2:<br/>
OpenCV是一个跨平台的计算机视觉库，最早由Intel开源得来。OpenCV发展的非常早，拥有众多的计算机视觉、数字图像处理和机器视觉等功能。OpenCV在功能上比Pillow更加强大很多，学习成本也高很多。<br/>
PIL：<br/>

### 1.3 torch
#### import torchvision.models as models
torchvision.models模块的子模块中包含以下模型结构。
* AlexNet
* VGG
* ResNet
* SqueezeNet
* DenseNet You can construct a model with random weights by calling its constructor:

#### import torchvision.transforms as transforms
对PIL.Image进行变换。<br/>
对Tensor进行变换。

#### import torchvision.datasets as datasets
torchvision.datasets中包含了以下数据集
* MNIST
* COCO（用于图像标注和目标检测）(Captioning and Detection)
* LSUN Classification
* ImageFolder
* Imagenet-12
* CIFAR10 and CIFAR100
* STL10
由于以上Datasets都是 torch.utils.data.Dataset的子类，所以，他们也可以通过torch.utils.data.DataLoader使用多线程（python的多进程）。

#### import torch.nn as nn


#### import torch.nn.functional as F
调用函数，包括Convolution函数，Pooling函数，非线性激活函数，Normalization函数，损失函数等。<br/>

#### import torch.optim as optim
是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性，使得未来能够集成更加复杂的方法。<br/>
为了使用torch.optim，你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

#### from torch.autograd import Variable
提供了类和函数用来对任意标量函数进行求导。<br/>要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的tensor包含进Variable对象中即可。

#### from torch.utils.data.dataset import Dataset



## 2 数据扩增
### 2.1 数据扩增目的
数据扩增可以增加训练集的样本，同时也可以有效缓解模型过拟合的情况，也可以给模型带来的更强的泛化能力。
### 2.2 数据扩增为什么有用
深度学习模型，具有多个隐藏层的神经网络，需要学习的参数非常非常多，而现实条件下，获取原始数据的数量相对而言比较少，如果不做扩增，容易造成过拟合。<br/>
其次，数据扩增可以增加样本空间。
### 2.3 数据扩增方法
在常见的数据扩增方法中，一般会从图像颜色、尺寸、形态、空间和像素等角度进行变换。当然不同的数据扩增方法可以自由进行组合，得到更加丰富的数据扩增方法。<br/>
以torchvision为例，常见的数据扩增方法包括：<br/>

* transforms.CenterCrop 对图片中心进行裁剪
* transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
* transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像
* transforms.Grayscale 对图像进行灰度变换
* transforms.Pad 使用固定值进行像素填充
* transforms.RandomAffine 随机仿射变换
* transforms.RandomCrop 随机区域裁剪
* transforms.RandomHorizontalFlip 随机水平翻转
* transforms.RandomRotation 随机旋转
* transforms.RandomVerticalFlip 随机垂直翻转




