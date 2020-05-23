# Task02：数据读取与数据扩增
## 1 Baseline导入包学习
### 1.1 pandas,numpy,os, sys, glob, shutil, json
常见的文件导入，读取，数据处理，科学计算等
### 1.2 cv2,PIL
cv2:<br/>
OpenCV是一个跨平台的计算机视觉库，最早由Intel开源得来。OpenCV发展的非常早，拥有众多的计算机视觉、数字图像处理和机器视觉等功能。OpenCV在功能上比Pillow更加强大很多，学习成本也高很多。
### 1.3 torch
**import torchvision.models as models** <br/>
torchvision.models    
模块的子模块中包含以下模型结构。
* AlexNet
* VGG
* ResNet
* SqueezeNet
* DenseNet You can construct a model with random weights by calling its constructor:

**import torchvision.transforms as transforms** <br/>
对PIL.Image进行变换。<br/>
对Tensor进行变换。

import torchvision.datasets as datasets <br/>
torchvision.datasets中包含了以下数据集
* MNIST
* COCO（用于图像标注和目标检测）(Captioning and Detection)
* LSUN Classification
* ImageFolder
* Imagenet-12
* CIFAR10 and CIFAR100
* STL10
由于以上Datasets都是 torch.utils.data.Dataset的子类，所以，他们也可以通过torch.utils.data.DataLoader使用多线程（python的多进程）。

import torch.nn as nn <br/>


import torch.nn.functional as F<br/>
调用函数，包括Convolution函数，Pooling函数，非线性激活函数，Normalization函数，损失函数等。<br/>

import torch.optim as optim <br/>
是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性，使得未来能够集成更加复杂的方法。<br/>
为了使用torch.optim，你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

from torch.autograd import Variable <br/>
提供了类和函数用来对任意标量函数进行求导。<br/>要想使用自动求导，只需要对已有的代码进行微小的改变。只需要将所有的tensor包含进Variable对象中即可。

from torch.utils.data.dataset import Dataset<br/>








