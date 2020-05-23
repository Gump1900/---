# Task02：数据读取与数据扩增
## 2 Baseline导入包学习
### 2.1 pandas,numpy,os, sys, glob, shutil, json
常见的文件导入，读取，数据处理，科学计算等
### 2.2 cv2,PIL
cv2:
### 2.3 torch
import torchvision.models as models


import torchvision.transforms as transforms 


import torchvision.datasets as datasets 


import torch.nn as nn 


import torch.nn.functional as F

import torch.optim as optim 
是一个实现了各种优化算法的库。大部分常用的方法得到支持，并且接口具备足够的通用性，使得未来能够集成更加复杂的方法。<br/>
为了使用torch.optim，你需要构建一个optimizer对象。这个对象能够保持当前参数状态并基于计算得到的梯度进行参数更新。

from torch.autograd import Variable 


from torch.utils.data.dataset  import Dataset






