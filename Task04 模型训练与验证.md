# Task04：模型训练与验证
## 1.模型训练目的
### 1.1 误差与过拟合
通常我们把学习器与与样本的真实输出之间的误差称之为误差，训练集上为经验误差，新样本上为泛化误差。<br/>
机器学习的目的，实际上是要在新样本上表现很好，因此，需要学习到样本当中的“普遍规律”，这样才能在遇到新样本上做出准确的判别。如果在训练集上，学习太好，可能把训练集上的独有特性也学习到了，那这种情况，称之为模型过拟合。与之相对应的是欠拟合，即学习器还未捕捉到完整的样本特征。<br/>
为了避免过拟合，通常，给定我们的数据集，可以通过“验证集”来对学习器的学习性能进行评估，因此需要划分一个验证集，且与训练集互斥，即验证集的样本没有在训练集上出现过。

## 2.模型训练方法
常见的数据集处理方法如下：
### 2.1 留出法
将数据集D划分为训练集S与验证集V，且S与V互斥，即D = S∪V，S∩V = Ф。<br/>
需要注意的内容：<br/>
* 训练集和验证集的划分，要尽可能保持数据分布的一致性。
* 单次使用结果可能不够可靠，一般需要多次划分取均值。

此外，如果划分时S太大，则由S得到的学习器则更接近于D训练出的模型，但是此时验证集太小，评估结果可能不够准确。反之，容易造成模型与由D得到的模型偏差较大，从而降低了评估结果的保真性。一般情况下，S的占比大概为 2/3~4/5。

### 2.2 交叉验证法
交叉验证法先将数据集划分为大小相同的k个子集，且子集两两互斥，每个子集都尽可能保证数据分布的一致性。然后，选择其中k-1个子集作为训练集S，剩下的1个子集作为验证集V，重复k次，得到的结果做平均，就是最终模型，称之为k折交叉验证。一般情况下，k取10。<br/>
与留出法类似，为了减小因样本划分不同而带来的偏差，k折交叉验证通常要随机使用不同的划分重复p次，最后一共进行了p×k次训练。常见的有10次10折交叉验证法。<br/>
当k等于D中样本数时，此时每一次的验证集仅有一个样本，我们称之为留一法，由于每次训练，仅有一个样本的不一样，使得得到的模型接近于D的模型，但是计算开销非常大。

### 2.3 自助法
上述两种生成验证集的方法，都有一个不足，即训练集大小总是少于给定样本数，这样会导致训练样本规模引起的偏差，因此，我们可以选择自助法(bootstrapping),每一次从数据集D中，有放回抽样m次，其中m为D的样本数，重新生成新的数据集D‘作为训练集，将D/D’作为验证集，根据计算可知，约有36.8%的样本数，可以用于验证。<br/>当数据集较小，难以划分训练集和验证集时，可以考虑使用自助法，且可以生成多个训练集，对于begging等集成学习有很好的帮助。而自助法的弊端，在于改变了原本D中的样本分布，因此，会引入估计偏差。

## 3.图像识别训练
本节我们继续使用Pytorch来完成模型的训练和验证。
首先，需要构造训练集和验证集。
```
# 生成训练集
train_path = glob.glob('D:/Data analysis practice/12 CV Competition/input/mchar_train/*.png')
train_path.sort()
train_json = json.load(open('D:/Data analysis practice/12 CV Competition/input/mchar_train.json'))
train_label = [train_json[x]['label'] for x in train_json]

train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=40, 
    shuffle=True, 
    num_workers=0,
)

# 生成验证集
val_path = glob.glob('D:/Data analysis practice/12 CV Competition/input/mchar_val/*.png')
val_path.sort()
val_json = json.load(open('D:/Data analysis practice/12 CV Competition/input/mchar_val.json'))
val_label = [val_json[x]['label'] for x in val_json]

val_loader = torch.utils.data.DataLoader(
    SVHNDataset(val_path, val_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=40, 
    shuffle=False, 
    num_workers=0,
)
```
每个Epoch训练代码和验证代码如下：
```
def train(train_loader, model, criterion, optimizer, epoch):
    # 切换模型为训练模式
    model.train()

    for i, (input, target) in enumerate(train_loader):
        c0, c1, c2, c3, c4, c5 = model(data[0])
        loss = criterion(c0, data[1][:, 0]) + \
                criterion(c1, data[1][:, 1]) + \
                criterion(c2, data[1][:, 2]) + \
                criterion(c3, data[1][:, 3]) + \
                criterion(c4, data[1][:, 4]) + \
                criterion(c5, data[1][:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            c0, c1, c2, c3, c4, c5 = model(data[0])
            loss = criterion(c0, data[1][:, 0]) + \
                    criterion(c1, data[1][:, 1]) + \
                    criterion(c2, data[1][:, 2]) + \
                    criterion(c3, data[1][:, 3]) + \
                    criterion(c4, data[1][:, 4]) + \
                    criterion(c5, data[1][:, 5])
            loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)
```
针对上面的一些名词，例如epoch，batch_size等，参考了一篇知乎文章[训练神经网络中最基本的三个概念：Epoch, Batch, Iteration](https://zhuanlan.zhihu.com/p/29409502)，而loss选择了交叉熵损失函数，老师课上提到的其他损失函数，还没来得及尝试。

# 4.模型保存
在Pytorch中模型的保存和加载非常简单，比较常见的做法是保存和加载模型参数：
`torch.save(model_object.state_dict(), 'model.pt')`<br/>
`model.load_state_dict(torch.load(' model.pt'))`

# 5.模型调参
跑一次baseline就要10个小时，实在是跑不动了。
