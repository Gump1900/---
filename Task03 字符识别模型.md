# Task03：字符识别模型
## 1.深度学习基础知识学习
本人入门机器学习和深度学习，理论部分是跟着台湾大学林轩田老师的机器学习基石和技法两学期课程自学的，林老师上课生动幽默，且理论基础十分扎实，在这里特别感谢林老师。且学习中参考了很多 AI有道 博主的内容，在这里也表示非常感谢！<br/>
虽然林老师的基石课程里，很少提到现在耳熟能详的SVM、Decesion Tree等模型，但是高屋建瓴地介绍了整个机器学习最为核心的内容，现在回看受益匪浅。以下是我的一些个人理解，如有不正确请指正，谢谢！
### 1.1 机器学习
林老师对机器学习的定义如下：Improving some performance measure with experience computed from data. 也就是机器从数据中总结经验，从数据中找出某种规律或者模型，并用它来解决实际问题。<br/>
对于一组数据，有输入x,输出y，机器学习的过程，就是根据先验知识选择模型，该模型对应的hypothesis set（用H表示），H中包含了许多不同的hypothesis， 通过演算法A，在训练样本D上进行训练，选择出一个最好的hypothesis，对应的函数表达式g就是我们最终要求的。一般情况下，g能最接近目标函数f。由此，引申出训练误差和泛化误差的概念，两分别在训练集和验证集上，描绘预测值与真实值之间的偏离程度。<br/>
从本质上说，机器学习建模的过程，就是追求泛化误差越小，但是实际上，泛化误差往往是得不到的，因为模型到实际运用中，总会有意想不到的数据出现。因此，要保证泛化误差尽可能小，必须找到一个代替品，且这个代替品可以有办法减小。我们采用训练误差来代替，也就是说，保证泛化误差尽可能小，必须满足2个条件：
* 训练误差≈泛化误差
* 训练误差→0

由这两个条件，可以看出来，第一个条件是保证模型抗数据扰动的能力强，即更换数据集，预测结果始终保持稳定；第二个条件刻画了模型本身在训练集的结果好坏。这两个条件，就对应着误差分解中的方差和偏差，前者不成立，模型可能过拟合，一点数据的变动都引起了误差的偏离过大，而后者不成立，说明模型欠拟合的可能较大，在训练集上连数据的一些特征都没有捕捉到。当然，以上讨论并未考虑噪音，这部分误差，刻画的本次学习问题的天然难易程度。

### 1.2 神经网络
神经网络一词，源于生物学。我们的大脑，每时每刻都需要处理海量信息，而大脑中有许许多多的神经元相互连接，负责处理传递信息。而人工智能方向的神经网络，便是参照生物体这一概念，通过一个又一个的“神经元”及他们连接而成的庞大网络，构成学习模型。而这里的神经元，则对应多种激活函数，如tanh，sigmoid，ReLU等等。<br/>
感知机由2层神经元组成，输入层对应输入数据，输出层是一个M-P神经元。若预测错误，则根据错误的程度做出调整。对于异或问题，单层神经元已不可用了。对于这类非线性可分问题，需要用到多层神经元，由此引出了：上一层与下一层全互连、没有同层连接、没有跨层连接的多层前馈神经网络（具体可以参考周志华老师的机器学习-第5章 神经网络）。此时，虽然模型的学习能力大大加强，但是参数非常多，这里的参数指的是每一层连接中的权重。由此引出著名的BackPropagation算法，简称BP算法。<br/>
BP算法的核心，在于使用链式法则，将误差与权重之间的关系，转化为各层级之间的关系。我们需要使用梯度下降法，求出误差极小值。因此，求导过程中，使用链式法则，得到权重和阈值的更新公式，逐步迭代，直到满足条件停止。
    
### 1.3 深度学习
这两天我在学习群里也与小伙伴们交流了一下，加上自己查阅资料，大致搞懂了一件事，深度学习，本质上也是机器学习的一种，都是构建模型，从海量数据中总结规律和经验。深度学习使用多个隐藏层的神经网络，构建学习模型。而我没想明白的是，深度学习为啥能取得这么好的学习效果？经小伙伴提醒，深度学习的可解释性，当前确实是一大问题，不像线性回归，决策树等模型通俗易懂。我自己再想了想，大致是神经网络中的激活函数和阈值设定，可以提取出一份数据中，影响最大的那部分数据，这其实就是特征。找出了所有的关键特征，我们的模型就可以在其他数据集上有较好的表现。
根据林老师-机器学习技法13课Deep Learning当中总结，深度学习较浅层神经网络，有几个需要克服的难点：
* （1）确认网络结构：这部分没有一个定值，更多的还是凭借经验和主观判断，比如说多少个隐藏层，每一层分别需要多少个神经元等。
* （2）很高的模型复杂度：模型复杂度过高，容易造成过拟合，这部分可以依靠当下海量数据来克服，而对待复杂模型，我们往往需要正则化。
* （3）参数优化：需要我们一开始谨慎初始化，避免出现局部最优值。可以用pre-training来实现。
* （4）计算量极大：海量数据加上高复杂度模型，训练时间会很长，可以通过GPU加速。<br/>
从上面对神经网络的介绍，深度学习最为重要的参数，就是权重。权重的本质，可以看作是一种特征变换，这部分类似于线性代数里面左乘矩阵，对向量进行空间变换。参考特征向量，仅仅只改变了向量的大小和正负向，并未使其发生偏转。因此，是否可以理解为，好的权重矩阵，也应当是保证输入的改变很小，尽可能地包含了绝大部分特征，这样进入下一层计算的时候，上一层的主要特征都能够得到保留，这样不断迭代，最终留下来的，应当都是我们需要的特征。所以权重的初始值就非常重要，如果一开始就遗漏了重要特征，后面的模型效果可能就不会太好。所以，进行pre-training找到较好的初始权重值就非常关键。<br/>
另一方面，海量的数据集也并非那么容易可以得到。在实际运用中，我们可以对原始数据进行处理，得到一系列不同的训练样本，扩充训练数量，比如说对图片进行旋转操作，这样既可以提取增加样本减少过拟合，也可以让模型更加专注于核心特征。<br/>
这两部分，对应的上面提到的（2）（3）两点，也是最难处理的两点。

## 2.CNN基础知识学习
### 2.1 CNN基本原理
CNN，全称Convolutional Neural Network，即卷积神经网络，是十分特殊的一类神经网络，在当前的图像识别、语义分割等方面十分可行。<br/>
这部分内容还在学习当中，参考了[Neural Network & Deep learning](http://neuralnetworksanddeeplearning.com/chap6.html)，后续根据自己理解再补充进来。

### 2.2 CNN发展过程
深度学习的主要模型有AlexNet、VGG、InceptionV3和ResNet的发展脉络，具体每部分内容的发展历史，如何推导，如何优化，后续再补充。

## 3.构建CNN模型
参考Baseline，以及第二次授课老师意见，更改了epoch为10，得到的结果是0.6，较第一次有明显提高，但是一次训练时间接近10个小时，实在是太长了。后面再参考老师意见，增加图片处理等部分，继续训练模型。
```
class SVHN_Model1(nn.Module):           #构建卷积神经网络
    def __init__(self):
        super(SVHN_Model1, self).__init__()
                
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
```
构造训练特征
```
def train(train_loader, model, criterion, optimizer,epoch):
    # 切换模型为训练模式
    model.train()
    train_loss = []
    
    for i, (input, target) in enumerate(train_loader):
        if use_cuda:
            input = input.cuda()
            target = target.cuda()
            
        c0, c1, c2, c3, c4 = model(input)
        target = target.long()
        
        loss = criterion(c0, target[:, 0]) + \           #定义损失函数
                criterion(c1, target[:, 1]) + \
                criterion(c2, target[:, 2]) + \
                criterion(c3, target[:, 3]) + \
                criterion(c4, target[:, 4])
        
        # loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 100 == 0:
            print(loss.item())
        
        train_loss.append(loss.item())
    return np.mean(train_loss)
```
验证集上做验证
```
def validate(val_loader, model, criterion):
    # 切换模型为预测模型
    model.eval()
    val_loss = []

    # 不记录模型梯度信息
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if use_cuda:
                input = input.cuda()
                target = target.cuda()
            
            c0, c1, c2, c3, c4 = model(input)
            target = target.long()
            loss = criterion(c0, target[:, 0]) + \          #定义损失函数
                    criterion(c1, target[:, 1]) + \
                    criterion(c2, target[:, 2]) + \
                    criterion(c3, target[:, 3]) + \
                    criterion(c4, target[:, 4])
            # loss /= 6
            val_loss.append(loss.item())
    return np.mean(val_loss)
```
计算训练误差和验证误差
```
model = SVHN_Model1()           #调用搭建好的CNN模型构建模型
criterion = nn.CrossEntropyLoss()     #使用交叉熵损失函数
optimizer = torch.optim.Adam(model.parameters(), 0.001)    #Pytorch参数优化
best_loss = 1000.0

use_cuda = False        #是否使用GPU加速计算
if use_cuda:
    model = model.cuda()

for epoch in range(10):      #对应不同的epoch，计算训练误差和验证误差
    train_loss = train(train_loader, model, criterion, optimizer,epoch)
    val_loss = validate(val_loader, model, criterion)
    
    val_label = [''.join(map(str, x)) for x in val_loader.dataset.img_label]
    val_predict_label = predict(val_loader, model, 1)
    val_predict_label = np.vstack([
        val_predict_label[:, :11].argmax(1),
        val_predict_label[:, 11:22].argmax(1),
        val_predict_label[:, 22:33].argmax(1),
        val_predict_label[:, 33:44].argmax(1),
        val_predict_label[:, 44:55].argmax(1),
    ]).T
    val_label_pred = []
    for x in val_predict_label:
        val_label_pred.append(''.join(map(str, x[x!=10])))
    
    val_char_acc = np.mean(np.array(val_label_pred) == np.array(val_label))
    
    print('Epoch: {0}, Train loss: {1} \t Val loss: {2}'.format(epoch, train_loss, val_loss))
    print(val_char_acc)
    # 记录下验证集精度
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), './model.pt')
```
得到的结果为：
```
12.320236206054688
4.686568260192871
3.117445468902588
3.428555727005005
3.1217775344848633
3.2852652072906494
2.8458895683288574
3.1151673793792725
Epoch: 0, Train loss: 3.493618921279907 	 Val loss: 3.565352352142334
0.3492
2.2881088256835938
1.959725260734558
2.607151746749878
1.951163649559021
1.7587155103683472
1.7689906358718872
2.092160940170288
2.278383493423462
Epoch: 1, Train loss: 2.188275608062744 	 Val loss: 3.167541268348694
0.4066
1.7606914043426514
1.748899221420288
1.755211353302002
1.750447392463684
1.5229179859161377
1.6405850648880005
1.966674566268921
1.4895097017288208
Epoch: 2, Train loss: 1.8201409175395966 	 Val loss: 2.8208281006813047
0.4747
1.6702637672424316
1.552194356918335
1.6906023025512695
1.655700922012329
1.592916488647461
2.108767032623291
2.273608922958374
1.7204703092575073
Epoch: 3, Train loss: 1.6197570888996125 	 Val loss: 2.7107202081680297
0.4957
1.2783684730529785
0.6630651950836182
1.2280359268188477
1.3281993865966797
0.7705426216125488
1.7711724042892456
1.0729039907455444
1.65019690990448
Epoch: 4, Train loss: 1.4680198764801025 	 Val loss: 2.7001162214279173
0.4976
1.250260591506958
1.4006667137145996
1.3169785737991333
0.9582787156105042
1.8479654788970947
1.639341950416565
1.5288372039794922
1.5343209505081177
Epoch: 5, Train loss: 1.3444286154508591 	 Val loss: 2.846746894836426
0.4763
0.9460337162017822
0.8201913237571716
1.1253620386123657
1.48394775390625
1.5830854177474976
1.5747222900390625
1.107652187347412
0.7611797451972961
Epoch: 6, Train loss: 1.2508838994503022 	 Val loss: 2.624796049833298
0.5273
1.0961517095565796
1.4375205039978027
0.8882400989532471
1.279791235923767
1.2786098718643188
0.8687078356742859
1.6031584739685059
1.1030560731887817
Epoch: 7, Train loss: 1.1649710725943248 	 Val loss: 2.5143720300197603
0.5259
1.1105328798294067
0.7317982316017151
1.074083685874939
1.0715482234954834
0.713513970375061
0.8418834209442139
1.3532941341400146
0.9909125566482544
Epoch: 8, Train loss: 1.0938419947226843 	 Val loss: 2.476309720516205
0.5553
0.8729658126831055
1.2555092573165894
0.7909500598907471
0.7385172247886658
1.8285408020019531
1.4075255393981934
0.6268761157989502
0.6235646605491638
Epoch: 9, Train loss: 1.0282455455462138 	 Val loss: 2.5279903419017793
0.5399
```
利用模型生成测试数据
```
def predict(test_loader, model, tta=10):
    model.eval()
    test_pred_tta = None
    
    # TTA 次数
    for _ in range(tta):
        test_pred = []
    
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                if use_cuda:
                    input = input.cuda()
                
                c0, c1, c2, c3, c4 = model(input)
                output = np.concatenate([
                    c0.data.numpy(), 
                    c1.data.numpy(),
                    c2.data.numpy(), 
                    c3.data.numpy(),
                    c4.data.numpy()], axis=1)
                test_pred.append(output)
        
        test_pred = np.vstack(test_pred)
        if test_pred_tta is None:
            test_pred_tta = test_pred
        else:
            test_pred_tta += test_pred
    
    return test_pred_tta
```

