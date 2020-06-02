# Task05 模型集成
## 1.机器学习模型集成
在机器学习中，常见的模型集成方法有：bagging,boosting,stacking，其中，非常具有代表性的算法分别是Random Forest（随机森林），GBDT（Gradient Boosting Decision Tree，梯度提升决策树），

## 2.深度学习模型集成
除了以上提到的机器学习集成方法，深度学习还有自己独特的集成方式。
### 2.1 Dropout
Dropout可以作为训练深度神经网络的一种技巧。在每个训练批次中，通过随机让一部分的节点停止工作。同时在预测的过程中让所有的节点都其作用。
```
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
       
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
        )
        
        self.fc0 = nn.Linear(32*3*7, 11)
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c0 = self.fc0(feat)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c0, c1, c2, c3, c4, c5
```

### 2.2 TTA(Test Time Augmentation)
测试集数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧，数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增，对同一个样本预测三次，然后对三次结果进行平均。
```
def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    # TTA 次数
    for _ in range(tta):
        test_pred = []
   
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            c0, c1, c2, c3, c4, c5 = model(input)
            output = np.concatenate([c0.data.numpy(), c1.data.numpy(),
                  c2.data.numpy(), c3.data.numpy(),
                  c4.data.numpy(), c5.data.numpy()], axis=1)
            test_pred.append(output)
       
    test_pred = np.vstack(test_pred)
    if test_pred_tta is None:
           test_pred_tta = test_pred
    else:
           test_pred_tta += test_pred
   
    return test_pred_tta
```
结果如下：<br/>








