# 机器学习纳米学位
走神司机 潘超 优达学城  
2018年6月6日
## I. 问题的定义
### 项目概述
项目将从一些车载视频摄像头中截取的静态图像识别驾驶员是否处于安全驾驶的状态。

生活中许多驾驶员喜欢一边开车一边做别的事情，如：打电话、发微信、吃东西、聊天、疲劳驾驶等等，安全隐患非常大。特别是一些大巴车司机，关乎到整个大巴车上几十个人的人身安全。该项目的数据来源方Kaggle比赛平台中的资料也指出：
> 根据美国疾病防控中心机动车安全部门的数据，五分之一的车祸是由分心的司机造成的。可悲的是，这意味着每年有42.5万人受伤，3000人因分心驾驶而死亡。

早期驾驶员状态检测方法主要是基于车辆运行状态的检测方法，包括车道偏离报警、转向盘检测等，对驾驶员本身的特征敏感度不高，容易因环境因素误判，也不能从根本上解决驾驶员状态检测的问题，而近年的基于深度学习的图像识别技术则提供了不错的解决办法，可以通过对视频图像进行分析检测驾驶员当前的状态并给予提醒，甚至在出现更严重的危险情况时通过车辆控制信号及时主动刹停汽车。

项目使用的数据源来源于二年前的Kaggle比赛，当年一共有1440名参赛队伍参于该赛事。
### 问题陈述
处理通过车载摄像头记录到的驾驶员状态图像，对图像进行识别处理，分析图像中驾驶员当前所处的状态，以满足对安全驾驶提醒的需求。需要从图像中识别包括如下的驾驶员状态：
0. 安全驾驶
1. 右手打字
2. 右手打电话
3. 左手打字
4. 左手打电话
5. 调收音机
6. 喝饮料
7. 拿后面的东西
8. 整理头发和化妆
9. 和其他乘客说话
每一张图片识别出的结果应该是该图片分别在十种状态中的概率值，如安全驾驶的图片的理想识别结果应该为c0类别的概率为1，其他9种类别的概率为0。

项目将使用卷积神经网络来识别这些图像属于哪种状态，卷积神经网络是从2012年开始迅速成长起来的新型图像识别算法和架构，至今已发展出许多不同的版本，在图像识别方面取得了越来越高的准确率。
### 评价指标
评估指标使用kaggle中该项目的评估方式，即multi-class logarithmic loss，损失值计算公式：
$$
logloss = -\frac{1}{N}\sum_{i=1}^{N}\sum_{j=1}^{M}y_{ij}log(p_{ij})
$$
公式中$N$为图像的数量，用于训练集时为当前训练集的数量，用于验证集时为验证集的数量，测试集同理。$M$表示图像标记的数量，在该项目中$M$为10。$y_{ij}$为第$i$个图像在第$j$分类中的标记概率，如果图像为该类，则该值为1，否则为0。$log$为自然对数，$p_{ij}$为第$i$个图像在第$j$分类中标记的预测概率。将每一张图像每个分类的预测概率的自然对数与分类目标标记的积相加再取负均值，最终即为多分类损失值。
## II. 分析
### 数据的探索
数据集来源于往年的Kaggle竞赛。数据集中包含大量车载摄像头对驾驶员位置的摄影截图，可清楚看到驾驶员的各种行为，包括打电话、喝饮料、拿后面的东西、打字等。数据集中将图片数据分为了训练集和测试集，训练集可用于该项目中训练模型，测试集可在模型训练完毕后检验预测效果，可提交至Kaggle中计算已训练模型的最终得分。训练集中已将图像标记分类，分为c0到c9一共十个文件夹存放，共22424张图片。测试集中有79729张未标记分类的图片。

数据集中每一张图片大小为640*480像素。图片中的驾驶员各种各样，有胖有瘦，有高有矮，有男有女、甚至还有不同肤色的驾驶员，有的驾驶员手臂上还有纹身。图片的光线有明，也有暗，甚至还有些有点爆光过度，导致难以发现手中的透明杯子。
1. 因光照原因看不见喝饮料的杯子  
![undefined](proposal_img/img_16.jpg)
2. 胖驾驶员  
![undefined](proposal_img/img_104.jpg)
3. 图像模糊
![undefined](proposal_img/img_316.jpg)
### 探索性可视化
训练数据中司机状态分类呈均匀分布：
![undefined](proposal_img/data.png)

在这一部分，你需要对数据的特征或特性进行概括性或提取性的可视化。这个可视化的过程应该要适应你所使用的数据。就你为何使用这个形式的可视化，以及这个可视化过程为什么是有意义的，进行一定的讨论。你需要考虑的问题：

你是否对数据中与问题有关的特性进行了可视化？
你对可视化结果进行详尽的分析和讨论了吗？
绘图的坐标轴，标题，基准面是不是清晰定义了？
### 算法和技术
> 介绍tensorflow、Keras
> 介绍数据增强
> 介绍InceptionV3
> 介绍Xception
> 在最后的报告中， 也需要介绍一下InceptionV3和Xception，讨论一下选择这两个算法的理由：训练参数少，速度快，提出了新的神经网络架构（网中网）

### 基准模型
使用Kaggle中该项目的排名分数做为基准模型。使用前10%的分数作为基准，第144名，最小损失值为0.25634。
## III. 方法
(大概 3-5 页）
### 数据预处理
> 对图像数据进行预处理：旋转、添加噪点、模糊、缩小图片

在这一部分， 你需要清晰记录你所有必要的数据预处理步骤。在前一个部分所描述的数据的异常或特性在这一部分需要被更正和处理。需要考虑的问题有：

如果你选择的算法需要进行特征选取或特征变换，你对此进行记录和描述了吗？
数据的探索这一部分中提及的异常和特性是否被更正了，对此进行记录和描述了吗？
如果你认为不需要进行预处理，你解释个中原因了吗？
### 执行过程
12. 第十二次
```
epochs = 20
batch_size=32
out_image_size = (299, 299)

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

sgd = SGD(lr=0.0001, decay=1e-7, momentum=0.9, nesterov=True)
```
```
Found 20622 images belonging to 10 classes.
Found 1802 images belonging to 10 classes.
model name: inceptionv3 , save weight file: saved_weights/inceptionv3_0.h5
Epoch 1/20
644/644 [==============================] - 823s 1s/step - loss: 2.3173 - acc: 0.1498 - val_loss: 2.0363 - val_acc: 0.3253
Epoch 2/20
644/644 [==============================] - 800s 1s/step - loss: 1.9181 - acc: 0.3263 - val_loss: 1.3049 - val_acc: 0.6217
Epoch 3/20
644/644 [==============================] - 799s 1s/step - loss: 1.2794 - acc: 0.5822 - val_loss: 0.7394 - val_acc: 0.7885
Epoch 4/20
644/644 [==============================] - 795s 1s/step - loss: 0.8195 - acc: 0.7442 - val_loss: 0.5157 - val_acc: 0.8387
Epoch 5/20
644/644 [==============================] - 794s 1s/step - loss: 0.5611 - acc: 0.8251 - val_loss: 0.3948 - val_acc: 0.8834
Epoch 6/20
644/644 [==============================] - 794s 1s/step - loss: 0.4305 - acc: 0.8688 - val_loss: 0.3637 - val_acc: 0.8761
Epoch 7/20
644/644 [==============================] - 795s 1s/step - loss: 0.3503 - acc: 0.8952 - val_loss: 0.3263 - val_acc: 0.8884
Epoch 8/20
644/644 [==============================] - 795s 1s/step - loss: 0.2964 - acc: 0.9087 - val_loss: 0.3658 - val_acc: 0.8705
```
![](report_img/model_loss_12.png)
15. 第十五次
```
epochs = 20
batch_size=32
out_image_size = (299, 299)

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

sgd = SGD(lr=0.0002, decay=6e-8, momentum=0.9, nesterov=True)
```
```
Found 20673 images belonging to 10 classes.
Found 1751 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_0.h5
Epoch 1/20
646/646 [==============================] - 816s 1s/step - loss: 2.1847 - acc: 0.2111 - val_loss: 1.7028 - val_acc: 0.4404
Epoch 2/20
646/646 [==============================] - 791s 1s/step - loss: 1.1735 - acc: 0.6189 - val_loss: 0.7388 - val_acc: 0.7847
Epoch 3/20
646/646 [==============================] - 789s 1s/step - loss: 0.5489 - acc: 0.8351 - val_loss: 0.4982 - val_acc: 0.8322
Epoch 4/20
646/646 [==============================] - 786s 1s/step - loss: 0.3462 - acc: 0.8973 - val_loss: 0.3818 - val_acc: 0.8814
Epoch 5/20
646/646 [==============================] - 786s 1s/step - loss: 0.2567 - acc: 0.9240 - val_loss: 0.3237 - val_acc: 0.8866
Epoch 6/20
646/646 [==============================] - 785s 1s/step - loss: 0.2097 - acc: 0.9383 - val_loss: 0.2726 - val_acc: 0.9201
Epoch 7/20
646/646 [==============================] - 786s 1s/step - loss: 0.1676 - acc: 0.9517 - val_loss: 0.3537 - val_acc: 0.8785
```
18. 第十九次
```
epochs = 20
batch_size=32
out_image_size = (299, 299)
val_loss_stop = 0.01

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(10, activation='softmax')(x)

op = Adam(lr=0.0003)
```
自动停止
```
Found 20600 images belonging to 10 classes.
Found 1824 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_0.h5
Epoch 1/10
643/643 [==============================] - 711s 1s/step - loss: 0.7969 - acc: 0.7280 - val_loss: 0.1855 - val_acc: 0.9276
Epoch 2/10
643/643 [==============================] - 695s 1s/step - loss: 0.3945 - acc: 0.8697 - val_loss: 0.2586 - val_acc: 0.9112

Found 20593 images belonging to 10 classes.
Found 1831 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_1.h5
Epoch 1/10
643/643 [==============================] - 705s 1s/step - loss: 0.8024 - acc: 0.7301 - val_loss: 0.2657 - val_acc: 0.9013
Epoch 2/10
643/643 [==============================] - 694s 1s/step - loss: 0.3988 - acc: 0.8697 - val_loss: 0.2169 - val_acc: 0.9227
Epoch 3/10
643/643 [==============================] - 698s 1s/step - loss: 0.3369 - acc: 0.8876 - val_loss: 0.2117 - val_acc: 0.9282

Found 20577 images belonging to 10 classes.
Found 1847 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_2.h5
Epoch 1/10
643/643 [==============================] - 700s 1s/step - loss: 0.7956 - acc: 0.7307 - val_loss: 0.7737 - val_acc: 0.7615
Epoch 2/10
643/643 [==============================] - 731s 1s/step - loss: 0.4041 - acc: 0.8697 - val_loss: 0.2663 - val_acc: 0.9265
Epoch 3/10
643/643 [==============================] - 729s 1s/step - loss: 0.3291 - acc: 0.8941 - val_loss: 0.2251 - val_acc: 0.9221
Epoch 4/10
643/643 [==============================] - 716s 1s/step - loss: 0.2981 - acc: 0.9040 - val_loss: 0.1864 - val_acc: 0.9293
Epoch 5/10
643/643 [==============================] - 706s 1s/step - loss: 0.2704 - acc: 0.9112 - val_loss: 0.2249 - val_acc: 0.9090

Found 20622 images belonging to 10 classes.
Found 1802 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_3.h5
Epoch 1/10
644/644 [==============================] - 725s 1s/step - loss: 0.7771 - acc: 0.7370 - val_loss: 0.3927 - val_acc: 0.8789
Epoch 2/10
644/644 [==============================] - 701s 1s/step - loss: 0.3955 - acc: 0.8714 - val_loss: 0.4207 - val_acc: 0.8605

Found 20665 images belonging to 10 classes.
Found 1759 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_4.h5
Epoch 1/10
645/645 [==============================] - 708s 1s/step - loss: 0.7724 - acc: 0.7365 - val_loss: 0.4658 - val_acc: 0.8461
Epoch 2/10
645/645 [==============================] - 707s 1s/step - loss: 0.3953 - acc: 0.8712 - val_loss: 0.5559 - val_acc: 0.8681

Found 20673 images belonging to 10 classes.
Found 1751 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_5.h5
Epoch 1/10
646/646 [==============================] - 717s 1s/step - loss: 0.7878 - acc: 0.7301 - val_loss: 0.3215 - val_acc: 0.9034
Epoch 2/10
646/646 [==============================] - 709s 1s/step - loss: 0.4031 - acc: 0.8661 - val_loss: 0.4694 - val_acc: 0.8634

Found 20714 images belonging to 10 classes.
Found 1710 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_6.h5
Epoch 1/10
647/647 [==============================] - 725s 1s/step - loss: 0.7579 - acc: 0.7437 - val_loss: 0.7232 - val_acc: 0.7995
Epoch 2/10
647/647 [==============================] - 710s 1s/step - loss: 0.3827 - acc: 0.8765 - val_loss: 1.0598 - val_acc: 0.7936

Found 20754 images belonging to 10 classes.
Found 1670 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_7.h5
Epoch 1/10
648/648 [==============================] - 735s 1s/step - loss: 0.7734 - acc: 0.7397 - val_loss: 0.2928 - val_acc: 0.9261
Epoch 2/10
648/648 [==============================] - 697s 1s/step - loss: 0.3890 - acc: 0.8727 - val_loss: 0.5392 - val_acc: 0.8504

Found 20740 images belonging to 10 classes.
Found 1684 images belonging to 10 classes.
model name: inceptionv3 , will save weight file: saved_weights/inceptionv3_8.h5
Epoch 1/10
648/648 [==============================] - 748s 1s/step - loss: 0.8334 - acc: 0.7189 - val_loss: 0.8631 - val_acc: 0.7386
Epoch 2/10
648/648 [==============================] - 707s 1s/step - loss: 0.4124 - acc: 0.8655 - val_loss: 0.6439 - val_acc: 0.8005
Epoch 3/10
648/648 [==============================] - 707s 1s/step - loss: 0.3281 - acc: 0.8932 - val_loss: 0.5077 - val_acc: 0.8155
Epoch 4/10
648/648 [==============================] - 709s 1s/step - loss: 0.2815 - acc: 0.9084 - val_loss: 0.4365 - val_acc: 0.8528
Epoch 5/10
648/648 [==============================] - 706s 1s/step - loss: 0.2655 - acc: 0.9141 - val_loss: 0.4702 - val_acc: 0.8714
```
19. 第十九次
使用Xception模型
```
epochs = 20
batch_size=32
out_image_size = (299, 299)
val_loss_stop = 0.01

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(10, activation='softmax')(x)

op = Adadelta()
```
```
Found 20754 images belonging to 10 classes.
Found 1670 images belonging to 10 classes.
model name: xception , will save weight file: saved_weights/xception_0.h5
Epoch 1/10
648/648 [==============================] - 738s 1s/step - loss: 0.6810 - acc: 0.7740 - val_loss: 0.3584 - val_acc: 0.8804
Epoch 2/10
648/648 [==============================] - 742s 1s/step - loss: 0.3443 - acc: 0.8896 - val_loss: 0.2012 - val_acc: 0.9417
Epoch 3/10
648/648 [==============================] - 736s 1s/step - loss: 0.2783 - acc: 0.9116 - val_loss: 0.2665 - val_acc: 0.9171
```
![](report_img/model_loss_19_0.png)
```
Found 20740 images belonging to 10 classes.
Found 1684 images belonging to 10 classes.
model name: xception , will save weight file: saved_weights/xception_1.h5
Epoch 1/10
648/648 [==============================] - 758s 1s/step - loss: 0.6857 - acc: 0.7733 - val_loss: 0.8786 - val_acc: 0.8017
Epoch 2/10
648/648 [==============================] - 727s 1s/step - loss: 0.3378 - acc: 0.8920 - val_loss: 0.2434 - val_acc: 0.9213
Epoch 3/10
648/648 [==============================] - 751s 1s/step - loss: 0.2795 - acc: 0.9109 - val_loss: 0.3092 - val_acc: 0.9099
```
![](report_img/model_loss_19_1.png)
```
Found 20762 images belonging to 10 classes.
Found 1662 images belonging to 10 classes.
model name: xception , will save weight file: saved_weights/xception_2.h5
Epoch 1/10
648/648 [==============================] - 747s 1s/step - loss: 0.6995 - acc: 0.7654 - val_loss: 0.1780 - val_acc: 0.9485
Epoch 2/10
648/648 [==============================] - 747s 1s/step - loss: 0.3354 - acc: 0.8927 - val_loss: 0.3984 - val_acc: 0.9001
```
![](report_img/model_loss_19_2.png)
```
Found 20769 images belonging to 10 classes.
Found 1655 images belonging to 10 classes.
model name: xception , will save weight file: saved_weights/xception_3.h5
Epoch 1/10
649/649 [==============================] - 763s 1s/step - loss: 0.6959 - acc: 0.7666 - val_loss: 0.5046 - val_acc: 0.8634
Epoch 2/10
649/649 [==============================] - 729s 1s/step - loss: 0.3423 - acc: 0.8892 - val_loss: 0.4905 - val_acc: 0.8315
Epoch 3/10
649/649 [==============================] - 743s 1s/step - loss: 0.2615 - acc: 0.9150 - val_loss: 0.4351 - val_acc: 0.8493
Epoch 4/10
649/649 [==============================] - 743s 1s/step - loss: 0.2368 - acc: 0.9225 - val_loss: 0.5228 - val_acc: 0.8505
```
![](report_img/model_loss_19_3.png)
```
Found 20778 images belonging to 10 classes.
Found 1646 images belonging to 10 classes.
model name: xception , will save weight file: saved_weights/xception_4.h5
Epoch 1/10
649/649 [==============================] - 781s 1s/step - loss: 0.6643 - acc: 0.7800 - val_loss: 1.0727 - val_acc: 0.7212
Epoch 2/10
649/649 [==============================] - 744s 1s/step - loss: 0.3314 - acc: 0.8915 - val_loss: 0.6295 - val_acc: 0.8499
Epoch 3/10
649/649 [==============================] - 731s 1s/step - loss: 0.2747 - acc: 0.9118 - val_loss: 0.5324 - val_acc: 0.8542
Epoch 4/10
649/649 [==============================] - 739s 1s/step - loss: 0.2272 - acc: 0.9279 - val_loss: 0.3692 - val_acc: 0.8811
Epoch 5/10
649/649 [==============================] - 743s 1s/step - loss: 0.2042 - acc: 0.9330 - val_loss: 0.7392 - val_acc: 0.8480
```
![](report_img/model_loss_19_4.png)
20. 第二十次
使用InceptionV3模型，KFold分为5组
```
epochs = 6
batch_size=32
out_image_size = (299, 299)
val_loss_stop = None

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

op = Adam(lr=0.0003, decay=3e-8)
```
```
Found 18017 images belonging to 10 classes.
Found 4407 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
WARNING:tensorflow:Variable *= will be deprecated. Use variable.assign_mul if you want assignment to the variable value or 'x = x * y' if you want a new python Tensor object.
Epoch 1/6
563/563 [==============================] - 925s 2s/step - loss: 0.9648 - acc: 0.6748 - val_loss: 0.6726 - val_acc: 0.8200
Epoch 2/6
563/563 [==============================] - 654s 1s/step - loss: 0.4614 - acc: 0.8518 - val_loss: 0.5315 - val_acc: 0.8369
Epoch 3/6
563/563 [==============================] - 652s 1s/step - loss: 0.3580 - acc: 0.8840 - val_loss: 0.7412 - val_acc: 0.8047
Epoch 4/6
563/563 [==============================] - 647s 1s/step - loss: 0.3204 - acc: 0.8978 - val_loss: 0.4037 - val_acc: 0.8652
Epoch 5/6
563/563 [==============================] - 645s 1s/step - loss: 0.2838 - acc: 0.9084 - val_loss: 0.4717 - val_acc: 0.8679
Epoch 6/6
563/563 [==============================] - 643s 1s/step - loss: 0.2694 - acc: 0.9142 - val_loss: 0.4441 - val_acc: 0.8807
```
![](report_img/model_loss_20_0.png)
21. 第二十一次
使用InceptionV3模型，KFold分为5组
```
epochs = 10
batch_size=32
out_image_size = (299, 299)
val_loss_stop = None

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
predictions = Dense(10, activation='softmax')(x)

op = Adam(lr=0.0001, decay=4e-8)
```
```
Found 18017 images belonging to 10 classes.
Found 4407 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
WARNING:tensorflow:Variable *= will be deprecated. Use variable.assign_mul if you want assignment to the variable value or 'x = x * y' if you want a new python Tensor object.
Epoch 1/10
563/563 [==============================] - 724s 1s/step - loss: 0.4831 - acc: 0.8466 - val_loss: 0.6032 - val_acc: 0.8159
Epoch 2/10
563/563 [==============================] - 494s 878ms/step - loss: 0.1116 - acc: 0.9647 - val_loss: 0.8054 - val_acc: 0.7876
Epoch 3/10
563/563 [==============================] - 489s 868ms/step - loss: 0.0741 - acc: 0.9772 - val_loss: 0.5017 - val_acc: 0.8485
Epoch 4/10
563/563 [==============================] - 483s 859ms/step - loss: 0.0597 - acc: 0.9805 - val_loss: 0.7995 - val_acc: 0.8417
```
22. 第二十二次
使用InceptionV3模型，KFold分为5组，使用ImageDataGenerator
```
epochs = 20
batch_size=64
out_image_size = (299, 299)
val_loss_stop = None

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

op = Adam(lr=0.0001, decay=10e-8)
```
```
Found 17949 images belonging to 10 classes.
Found 4475 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
Epoch 1/20
280/280 [==============================] - 506s 2s/step - loss: 0.7758 - acc: 0.7400 - val_loss: 0.5572 - val_acc: 0.8306
Epoch 2/20
280/280 [==============================] - 505s 2s/step - loss: 0.1233 - acc: 0.9615 - val_loss: 0.6580 - val_acc: 0.7973
Epoch 3/20
280/280 [==============================] - 484s 2s/step - loss: 0.0745 - acc: 0.9772 - val_loss: 0.5475 - val_acc: 0.8243
Epoch 4/20
280/280 [==============================] - 498s 2s/step - loss: 0.0531 - acc: 0.9845 - val_loss: 1.4999 - val_acc: 0.6440
Epoch 5/20
280/280 [==============================] - 489s 2s/step - loss: 0.0415 - acc: 0.9864 - val_loss: 0.5675 - val_acc: 0.8376
Epoch 6/20
280/280 [==============================] - 485s 2s/step - loss: 0.0399 - acc: 0.9883 - val_loss: 1.0448 - val_acc: 0.7321
Epoch 7/20
280/280 [==============================] - 481s 2s/step - loss: 0.0345 - acc: 0.9887 - val_loss: 1.1757 - val_acc: 0.7285
Epoch 8/20
280/280 [==============================] - 487s 2s/step - loss: 0.0329 - acc: 0.9909 - val_loss: 0.6601 - val_acc: 0.8471
Epoch 9/20
280/280 [==============================] - 490s 2s/step - loss: 0.0246 - acc: 0.9922 - val_loss: 0.8080 - val_acc: 0.8105
Epoch 10/20
280/280 [==============================] - 488s 2s/step - loss: 0.0223 - acc: 0.9933 - val_loss: 0.9342 - val_acc: 0.7991
Epoch 11/20
280/280 [==============================] - 485s 2s/step - loss: 0.0247 - acc: 0.9929 - val_loss: 0.5744 - val_acc: 0.8585
Epoch 12/20
280/280 [==============================] - 482s 2s/step - loss: 0.0226 - acc: 0.9928 - val_loss: 0.6600 - val_acc: 0.8261
Epoch 13/20
280/280 [==============================] - 496s 2s/step - loss: 0.0211 - acc: 0.9929 - val_loss: 0.3713 - val_acc: 0.9049
Epoch 14/20
280/280 [==============================] - 481s 2s/step - loss: 0.0214 - acc: 0.9935 - val_loss: 0.6793 - val_acc: 0.8179
Epoch 15/20
280/280 [==============================] - 483s 2s/step - loss: 0.0240 - acc: 0.9927 - val_loss: 0.4862 - val_acc: 0.8845
Epoch 16/20
280/280 [==============================] - 482s 2s/step - loss: 0.0173 - acc: 0.9949 - val_loss: 1.3515 - val_acc: 0.7095
Epoch 17/20
280/280 [==============================] - 473s 2s/step - loss: 0.0160 - acc: 0.9959 - val_loss: 0.4426 - val_acc: 0.8834
```
24. 第二十四次
使用InceptionV3模型，KFold分为5组，使用ImageDataGenerator，不锁层
```
epochs = 20
batch_size=64
out_image_size = (299, 299)
val_loss_stop = None

x = Dropout(0.5)(x)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

op = SGD(lr=0.0002, decay=4e-8, momentum=0.9, nesterov=True)
```
```
Found 18060 images belonging to 10 classes.
Found 4364 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
Epoch 1/20
282/282 [==============================] - 487s 2s/step - loss: 2.3355 - acc: 0.1451 - val_loss: 2.1266 - val_acc: 0.2877
Epoch 2/20
282/282 [==============================] - 492s 2s/step - loss: 2.0175 - acc: 0.2914 - val_loss: 1.7082 - val_acc: 0.4596
Epoch 3/20
282/282 [==============================] - 485s 2s/step - loss: 1.4400 - acc: 0.5394 - val_loss: 1.1838 - val_acc: 0.6365
Epoch 4/20
282/282 [==============================] - 490s 2s/step - loss: 0.8942 - acc: 0.7267 - val_loss: 0.9241 - val_acc: 0.6990
Epoch 5/20
282/282 [==============================] - 489s 2s/step - loss: 0.5983 - acc: 0.8210 - val_loss: 0.7710 - val_acc: 0.7390
Epoch 6/20
282/282 [==============================] - 480s 2s/step - loss: 0.4437 - acc: 0.8670 - val_loss: 0.7111 - val_acc: 0.7569
Epoch 7/20
282/282 [==============================] - 482s 2s/step - loss: 0.3534 - acc: 0.8965 - val_loss: 0.6931 - val_acc: 0.7624
Epoch 8/20
282/282 [==============================] - 494s 2s/step - loss: 0.2924 - acc: 0.9151 - val_loss: 0.6475 - val_acc: 0.7721
Epoch 9/20
282/282 [==============================] - 479s 2s/step - loss: 0.2515 - acc: 0.9249 - val_loss: 0.6469 - val_acc: 0.7769
Epoch 10/20
282/282 [==============================] - 484s 2s/step - loss: 0.2182 - acc: 0.9365 - val_loss: 0.6128 - val_acc: 0.7916
Epoch 11/20
282/282 [==============================] - 487s 2s/step - loss: 0.2000 - acc: 0.9413 - val_loss: 0.6187 - val_acc: 0.7920
Epoch 12/20
282/282 [==============================] - 484s 2s/step - loss: 0.1724 - acc: 0.9500 - val_loss: 0.6657 - val_acc: 0.7790
```
26. 第二十六次
使用InceptionV3模型，KFold分为5组，使用ImageDataGenerator，不锁层
```
epochs = 30
batch_size=96
out_image_size = (299, 299)
val_loss_stop = None

x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

op = SGD(lr=0.0003, decay=9e-8, momentum=0.9, nesterov=True)
```
```
Found 17949 images belonging to 10 classes.
Found 4475 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
Epoch 1/30
186/186 [==============================] - 498s 3s/step - loss: 2.3412 - acc: 0.1413 - val_loss: 2.1325 - val_acc: 0.2434
Epoch 2/30
186/186 [==============================] - 478s 3s/step - loss: 2.0382 - acc: 0.2836 - val_loss: 1.7160 - val_acc: 0.4688
Epoch 3/30
186/186 [==============================] - 507s 3s/step - loss: 1.4908 - acc: 0.5310 - val_loss: 1.1032 - val_acc: 0.6166
Epoch 4/30
186/186 [==============================] - 483s 3s/step - loss: 0.9380 - acc: 0.7236 - val_loss: 0.7529 - val_acc: 0.7568
Epoch 5/30
186/186 [==============================] - 486s 3s/step - loss: 0.6222 - acc: 0.8237 - val_loss: 0.5732 - val_acc: 0.8238
Epoch 6/30
186/186 [==============================] - 484s 3s/step - loss: 0.4437 - acc: 0.8760 - val_loss: 0.4956 - val_acc: 0.8478
Epoch 7/30
186/186 [==============================] - 490s 3s/step - loss: 0.3438 - acc: 0.9016 - val_loss: 0.4629 - val_acc: 0.8598
Epoch 8/30
186/186 [==============================] - 482s 3s/step - loss: 0.2861 - acc: 0.9178 - val_loss: 0.4212 - val_acc: 0.8732
Epoch 9/30
186/186 [==============================] - 491s 3s/step - loss: 0.2404 - acc: 0.9304 - val_loss: 0.4211 - val_acc: 0.8723
Epoch 10/30
186/186 [==============================] - 489s 3s/step - loss: 0.2125 - acc: 0.9389 - val_loss: 0.4176 - val_acc: 0.8730
Epoch 11/30
186/186 [==============================] - 491s 3s/step - loss: 0.1875 - acc: 0.9464 - val_loss: 0.4217 - val_acc: 0.8798
```
27. 第二十七次
使用InceptionV3模型，KFold分为5组，使用ImageDataGenerator，不锁层
```
epochs = 30
batch_size=96
out_image_size = (299, 299)
val_loss_stop = 0.01

x = GlobalAveragePooling2D()(x)
x = Dropout(0.8)(x)
predictions = Dense(10, activation='softmax')(x)

op = SGD(lr=0.0003, decay=9e-8, momentum=0.9, nesterov=True)
```
```
Found 18017 images belonging to 10 classes.
Found 4407 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
WARNING:tensorflow:Variable *= will be deprecated. Use variable.assign_mul if you want assignment to the variable value or 'x = x * y' if you want a new python Tensor object.
Epoch 1/30
187/187 [==============================] - 500s 3s/step - loss: 2.4817 - acc: 0.1111 - val_loss: 2.2818 - val_acc: 0.1481
Epoch 2/30
187/187 [==============================] - 496s 3s/step - loss: 2.3047 - acc: 0.1266 - val_loss: 2.2683 - val_acc: 0.1977
Epoch 3/30
187/187 [==============================] - 485s 3s/step - loss: 2.2719 - acc: 0.1487 - val_loss: 2.2364 - val_acc: 0.2873
Epoch 4/30
187/187 [==============================] - 489s 3s/step - loss: 2.1899 - acc: 0.2025 - val_loss: 2.1157 - val_acc: 0.3275
Epoch 5/30
187/187 [==============================] - 483s 3s/step - loss: 1.8973 - acc: 0.3404 - val_loss: 1.7119 - val_acc: 0.4139
Epoch 6/30
187/187 [==============================] - 493s 3s/step - loss: 1.3464 - acc: 0.5550 - val_loss: 1.3021 - val_acc: 0.5442
Epoch 7/30
187/187 [==============================] - 487s 3s/step - loss: 0.9071 - acc: 0.7056 - val_loss: 1.1667 - val_acc: 0.5958
Epoch 8/30
187/187 [==============================] - 485s 3s/step - loss: 0.6448 - acc: 0.8039 - val_loss: 1.0653 - val_acc: 0.6512
Epoch 9/30
187/187 [==============================] - 468s 3s/step - loss: 0.4673 - acc: 0.8652 - val_loss: 0.9697 - val_acc: 0.6826
Epoch 10/30
187/187 [==============================] - 482s 3s/step - loss: 0.3591 - acc: 0.8984 - val_loss: 0.9909 - val_acc: 0.6951
```
![](report_img/model_loss_27_0.png)
28. 第二十八次
使用InceptionV3模型，KFold分为5组，使用ImageDataGenerator，从249开始锁层
```
epochs = 30
batch_size=96
out_image_size = (299, 299)
val_loss_stop = 0.01

x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(10, activation='softmax')(x)

op = SGD(lr=0.0003, decay=9e-8, momentum=0.9, nesterov=True)
```
```
Found 17949 images belonging to 10 classes.
Found 4475 images belonging to 10 classes.
model name: inception_v3 , will save weight file: saved_weights/inception_v3_0.h5
Epoch 1/30
186/186 [==============================] - 463s 2s/step - loss: 2.3617 - acc: 0.1329 - val_loss: 2.4160 - val_acc: 0.1039
Epoch 2/30
186/186 [==============================] - 460s 2s/step - loss: 2.1765 - acc: 0.2169 - val_loss: 2.4193 - val_acc: 0.1037
Epoch 3/30
186/186 [==============================] - 461s 2s/step - loss: 1.9887 - acc: 0.3042 - val_loss: 2.4258 - val_acc: 0.1037
```
**29. 29**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- val_loss_stop: 0.01
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(10, activation='softmax')(x)
  ```
- 优化器: SGD
  - lr = 0.0003
  - decay = 9e-8

**结果：**
```
Epoch 1/30
187/187 [==============================] - 821s 4s/step - loss: 2.3334 - acc: 0.1457 - val_loss: 2.1769 - val_acc: 0.2509
Epoch 2/30
187/187 [==============================] - 479s 3s/step - loss: 2.0047 - acc: 0.2973 - val_loss: 1.8025 - val_acc: 0.3958
Epoch 3/30
187/187 [==============================] - 476s 3s/step - loss: 1.4364 - acc: 0.5514 - val_loss: 1.2536 - val_acc: 0.5958
Epoch 4/30
187/187 [==============================] - 491s 3s/step - loss: 0.9082 - acc: 0.7324 - val_loss: 0.9935 - val_acc: 0.6595
Epoch 5/30
187/187 [==============================] - 489s 3s/step - loss: 0.6038 - acc: 0.8266 - val_loss: 0.8646 - val_acc: 0.7074
Epoch 6/30
187/187 [==============================] - 491s 3s/step - loss: 0.4371 - acc: 0.8767 - val_loss: 0.8365 - val_acc: 0.7106
Epoch 7/30
187/187 [==============================] - 480s 3s/step - loss: 0.3423 - acc: 0.9041 - val_loss: 0.7979 - val_acc: 0.7248
Epoch 8/30
187/187 [==============================] - 485s 3s/step - loss: 0.2744 - acc: 0.9215 - val_loss: 0.8147 - val_acc: 0.7278
```
![](report_img/model_loss_29_0.png)
**说明：**
模型在第3代时val_loss下降缓慢，并在第8代时出现过拟合
**30. 30**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- val_loss_stop: 0.01
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.3)(x)
  predictions = Dense(10, activation='softmax')(x)
  ```
- 优化器: SGD
  - lr = 0.0003
  - decay = 9e-8

**结果：**
```
Epoch 1/30
188/188 [==============================] - 485s 3s/step - loss: 2.2476 - acc: 0.1738 - val_loss: 2.1018 - val_acc: 0.2347
Epoch 2/30
188/188 [==============================] - 488s 3s/step - loss: 1.8626 - acc: 0.4055 - val_loss: 1.5970 - val_acc: 0.5264
Epoch 3/30
188/188 [==============================] - 486s 3s/step - loss: 1.2457 - acc: 0.6573 - val_loss: 1.1073 - val_acc: 0.6343
Epoch 4/30
188/188 [==============================] - 482s 3s/step - loss: 0.7851 - acc: 0.7897 - val_loss: 0.9541 - val_acc: 0.6949
Epoch 5/30
188/188 [==============================] - 493s 3s/step - loss: 0.5371 - acc: 0.8531 - val_loss: 0.9079 - val_acc: 0.6903
Epoch 6/30
188/188 [==============================] - 478s 3s/step - loss: 0.3935 - acc: 0.8922 - val_loss: 0.8682 - val_acc: 0.7162
Epoch 7/30
188/188 [==============================] - 482s 3s/step - loss: 0.3086 - acc: 0.9154 - val_loss: 0.8218 - val_acc: 0.7502
Epoch 8/30
188/188 [==============================] - 486s 3s/step - loss: 0.2552 - acc: 0.9296 - val_loss: 0.8038 - val_acc: 0.7493
Epoch 9/30
188/188 [==============================] - 484s 3s/step - loss: 0.2178 - acc: 0.9399 - val_loss: 0.8054 - val_acc: 0.7535
```
![](report_img/model_loss_30_0.png)
**说明：**
尝试降低Dropout比例为0.3，第9代出现轻微上扬，loss下降太快，val_loss出现轻微过拟合现象
**31. 31**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- 停止提升轮数(patience): 2
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.7)(x)
  predictions = Dense(10, activation='softmax')(x)
  ```
- 优化器: SGD
  - lr = 0.0003
  - decay = 9e-8

**结果：**
```
Epoch 1/30
187/187 [==============================] - 490s 3s/step - loss: 2.4303 - acc: 0.1169 - val_loss: 2.2587 - val_acc: 0.1694
Epoch 2/30
187/187 [==============================] - 473s 3s/step - loss: 2.2521 - acc: 0.1708 - val_loss: 2.1665 - val_acc: 0.2917
Epoch 3/30
187/187 [==============================] - 483s 3s/step - loss: 2.0116 - acc: 0.2949 - val_loss: 1.8511 - val_acc: 0.3868
Epoch 4/30
187/187 [==============================] - 475s 3s/step - loss: 1.4214 - acc: 0.5359 - val_loss: 1.3291 - val_acc: 0.5134
Epoch 5/30
187/187 [==============================] - 491s 3s/step - loss: 0.8565 - acc: 0.7343 - val_loss: 1.1148 - val_acc: 0.5900
Epoch 6/30
187/187 [==============================] - 471s 3s/step - loss: 0.5490 - acc: 0.8373 - val_loss: 0.9959 - val_acc: 0.6444
Epoch 7/30
187/187 [==============================] - 483s 3s/step - loss: 0.3970 - acc: 0.8857 - val_loss: 0.9279 - val_acc: 0.6803
Epoch 8/30
187/187 [==============================] - 482s 3s/step - loss: 0.3017 - acc: 0.9143 - val_loss: 0.8721 - val_acc: 0.6970
Epoch 9/30
187/187 [==============================] - 476s 3s/step - loss: 0.2547 - acc: 0.9263 - val_loss: 0.9392 - val_acc: 0.6972
Epoch 10/30
187/187 [==============================] - 468s 3s/step - loss: 0.2064 - acc: 0.9398 - val_loss: 0.8845 - val_acc: 0.7118
```
![](report_img/model_loss_31_0.png)
**说明：**
尝试升高Dropout比例为0.7，loss和val_loss下降速度慢，且同样val_loss在0.8左右出现过拟合现象
**32. 32**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- 停止提升轮数(patience): 2
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(10, activation='softmax')(x)
  ```
- 优化器: SGD
  - lr = 0.0002
  - decay = 9e-8

**结果：**
```
Epoch 1/30
188/188 [==============================] - 496s 3s/step - loss: 2.3805 - acc: 0.1293 - val_loss: 2.1918 - val_acc: 0.2431
Epoch 2/30
188/188 [==============================] - 476s 3s/step - loss: 2.1874 - acc: 0.2112 - val_loss: 1.9785 - val_acc: 0.3947
Epoch 3/30
188/188 [==============================] - 493s 3s/step - loss: 1.9071 - acc: 0.3509 - val_loss: 1.5750 - val_acc: 0.5248
Epoch 4/30
188/188 [==============================] - 484s 3s/step - loss: 1.4455 - acc: 0.5495 - val_loss: 1.1933 - val_acc: 0.6405
Epoch 5/30
188/188 [==============================] - 492s 3s/step - loss: 1.0245 - acc: 0.6946 - val_loss: 0.9980 - val_acc: 0.6850
Epoch 6/30
188/188 [==============================] - 482s 3s/step - loss: 0.7428 - acc: 0.7849 - val_loss: 0.8743 - val_acc: 0.7275
Epoch 7/30
188/188 [==============================] - 491s 3s/step - loss: 0.5806 - acc: 0.8321 - val_loss: 0.8376 - val_acc: 0.7361
Epoch 8/30
188/188 [==============================] - 476s 3s/step - loss: 0.4589 - acc: 0.8698 - val_loss: 0.8047 - val_acc: 0.7428
Epoch 9/30
188/188 [==============================] - 477s 3s/step - loss: 0.3871 - acc: 0.8892 - val_loss: 0.7625 - val_acc: 0.7706
Epoch 10/30
188/188 [==============================] - 472s 3s/step - loss: 0.3410 - acc: 0.9008 - val_loss: 0.7680 - val_acc: 0.7711
Epoch 11/30
188/188 [==============================] - 474s 3s/step - loss: 0.2909 - acc: 0.9163 - val_loss: 0.7543 - val_acc: 0.7725
Epoch 12/30
188/188 [==============================] - 484s 3s/step - loss: 0.2665 - acc: 0.9203 - val_loss: 0.7452 - val_acc: 0.7708
Epoch 13/30
188/188 [==============================] - 477s 3s/step - loss: 0.2337 - acc: 0.9323 - val_loss: 0.7198 - val_acc: 0.7833
Epoch 14/30
188/188 [==============================] - 484s 3s/step - loss: 0.2197 - acc: 0.9364 - val_loss: 0.7321 - val_acc: 0.7803
Epoch 15/30
188/188 [==============================] - 477s 3s/step - loss: 0.2003 - acc: 0.9414 - val_loss: 0.7250 - val_acc: 0.7861
```
![](report_img/model_loss_32_0.png)
**说明：**
Dropout恢复为0.5，尝试通过使用减低学习率来减少过拟合，学习率降为0.0002。
**33. 33**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- 停止提升轮数(patience): 5
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(10, activation='softmax')(x)
  ```
- 优化器: SGD
  - lr = 0.0001
  - decay = 9e-8

**结果：**
```
Epoch 1/30
186/186 [==============================] - 491s 3s/step - loss: 2.4127 - acc: 0.1180 - val_loss: 2.2753 - val_acc: 0.1209
Epoch 2/30
186/186 [==============================] - 480s 3s/step - loss: 2.3112 - acc: 0.1495 - val_loss: 2.2073 - val_acc: 0.1680
Epoch 3/30
186/186 [==============================] - 486s 3s/step - loss: 2.2089 - acc: 0.1970 - val_loss: 2.1102 - val_acc: 0.2396
Epoch 4/30
186/186 [==============================] - 482s 3s/step - loss: 2.0815 - acc: 0.2645 - val_loss: 1.9669 - val_acc: 0.3530
Epoch 5/30
186/186 [==============================] - 487s 3s/step - loss: 1.9131 - acc: 0.3430 - val_loss: 1.7733 - val_acc: 0.4975
Epoch 6/30
186/186 [==============================] - 477s 3s/step - loss: 1.6889 - acc: 0.4400 - val_loss: 1.5697 - val_acc: 0.5772
Epoch 7/30
186/186 [==============================] - 487s 3s/step - loss: 1.4547 - acc: 0.5384 - val_loss: 1.4068 - val_acc: 0.6313
Epoch 8/30
186/186 [==============================] - 480s 3s/step - loss: 1.2337 - acc: 0.6176 - val_loss: 1.2480 - val_acc: 0.6653
Epoch 9/30
186/186 [==============================] - 484s 3s/step - loss: 1.0535 - acc: 0.6694 - val_loss: 1.1291 - val_acc: 0.6827
Epoch 10/30
186/186 [==============================] - 483s 3s/step - loss: 0.8924 - acc: 0.7289 - val_loss: 1.0438 - val_acc: 0.7047
Epoch 11/30
186/186 [==============================] - 475s 3s/step - loss: 0.7840 - acc: 0.7604 - val_loss: 0.9671 - val_acc: 0.7156
Epoch 12/30
186/186 [==============================] - 494s 3s/step - loss: 0.6780 - acc: 0.7952 - val_loss: 0.9007 - val_acc: 0.7285
Epoch 13/30
186/186 [==============================] - 474s 3s/step - loss: 0.6001 - acc: 0.8188 - val_loss: 0.8616 - val_acc: 0.7348
```
**说明：**
学习率降为0.0001，非常慢，训练未结束

**34. 34**
**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- 停止提升轮数(patience): 5
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(10, activation='softmax')(x)
  ```
- 优化器: SGD
  - lr = 0.0002
  - decay = 20e-8

**结果：**
```
Epoch 1/30
187/187 [==============================] - 808s 4s/step - loss: 2.3717 - acc: 0.1260 - val_loss: 2.1831 - val_acc: 0.2176
Epoch 2/30
187/187 [==============================] - 506s 3s/step - loss: 2.1638 - acc: 0.2171 - val_loss: 2.0203 - val_acc: 0.3153
Epoch 3/30
187/187 [==============================] - 491s 3s/step - loss: 1.8767 - acc: 0.3629 - val_loss: 1.7501 - val_acc: 0.4634
Epoch 4/30
187/187 [==============================] - 485s 3s/step - loss: 1.4680 - acc: 0.5393 - val_loss: 1.3558 - val_acc: 0.5674
Epoch 5/30
187/187 [==============================] - 486s 3s/step - loss: 1.0615 - acc: 0.6856 - val_loss: 1.1250 - val_acc: 0.6241
Epoch 6/30
187/187 [==============================] - 483s 3s/step - loss: 0.7588 - acc: 0.7848 - val_loss: 0.9835 - val_acc: 0.6734
Epoch 7/30
187/187 [==============================] - 488s 3s/step - loss: 0.5683 - acc: 0.8359 - val_loss: 0.9091 - val_acc: 0.7042
Epoch 8/30
187/187 [==============================] - 489s 3s/step - loss: 0.4445 - acc: 0.8762 - val_loss: 0.8566 - val_acc: 0.7280
Epoch 9/30
187/187 [==============================] - 484s 3s/step - loss: 0.3688 - acc: 0.8968 - val_loss: 0.8086 - val_acc: 0.7407
Epoch 10/30
187/187 [==============================] - 471s 3s/step - loss: 0.3182 - acc: 0.9114 - val_loss: 0.8428 - val_acc: 0.7326
Epoch 11/30
187/187 [==============================] - 490s 3s/step - loss: 0.2764 - acc: 0.9211 - val_loss: 0.7950 - val_acc: 0.7488
Epoch 12/30
187/187 [==============================] - 482s 3s/step - loss: 0.2400 - acc: 0.9327 - val_loss: 0.7864 - val_acc: 0.7535
Epoch 13/30
187/187 [==============================] - 481s 3s/step - loss: 0.2183 - acc: 0.9371 - val_loss: 0.8432 - val_acc: 0.7451
Epoch 14/30
187/187 [==============================] - 467s 2s/step - loss: 0.1985 - acc: 0.9428 - val_loss: 0.8217 - val_acc: 0.7521
Epoch 15/30
187/187 [==============================] - 482s 3s/step - loss: 0.1843 - acc: 0.9478 - val_loss: 0.8261 - val_acc: 0.7514
```
**说明：**
学习率为0.0001时loss下降太慢，尝试提高学习率，并提高衰减率，看是否可达到相同效果
**35. 35**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- 停止提升轮数(patience): 3
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(10, activation='softmax', use_bias=False)(x)
  ```
- 优化器: SGD
  - lr = 0.0003
  - decay = 30e-8

**结果：**
```
Epoch 1/30
186/186 [==============================] - 526s 3s/step - loss: 2.3387 - acc: 0.1413 - val_loss: 2.1858 - val_acc: 0.2312
Epoch 2/30
186/186 [==============================] - 488s 3s/step - loss: 1.9921 - acc: 0.3061 - val_loss: 1.7949 - val_acc: 0.4606
Epoch 3/30
186/186 [==============================] - 509s 3s/step - loss: 1.3871 - acc: 0.5684 - val_loss: 1.2660 - val_acc: 0.6409
Epoch 4/30
186/186 [==============================] - 488s 3s/step - loss: 0.8625 - acc: 0.7415 - val_loss: 0.9622 - val_acc: 0.7147
Epoch 5/30
186/186 [==============================] - 505s 3s/step - loss: 0.5828 - acc: 0.8245 - val_loss: 0.8393 - val_acc: 0.7457
Epoch 6/30
186/186 [==============================] - 531s 3s/step - loss: 0.4346 - acc: 0.8738 - val_loss: 0.7152 - val_acc: 0.7822
Epoch 7/30
186/186 [==============================] - 522s 3s/step - loss: 0.3379 - acc: 0.9036 - val_loss: 0.6595 - val_acc: 0.7912
Epoch 8/30
186/186 [==============================] - 511s 3s/step - loss: 0.2794 - acc: 0.9206 - val_loss: 0.6053 - val_acc: 0.8177
Epoch 9/30
186/186 [==============================] - 515s 3s/step - loss: 0.2371 - acc: 0.9326 - val_loss: 0.5575 - val_acc: 0.8286
Epoch 10/30
186/186 [==============================] - 528s 3s/step - loss: 0.2014 - acc: 0.9430 - val_loss: 0.5449 - val_acc: 0.8331
Epoch 11/30
186/186 [==============================] - 515s 3s/step - loss: 0.1813 - acc: 0.9491 - val_loss: 0.5305 - val_acc: 0.8361
Epoch 12/30
186/186 [==============================] - 512s 3s/step - loss: 0.1583 - acc: 0.9549 - val_loss: 0.5480 - val_acc: 0.8272
Epoch 13/30
186/186 [==============================] - 507s 3s/step - loss: 0.1432 - acc: 0.9601 - val_loss: 0.5344 - val_acc: 0.8338
Epoch 14/30
186/186 [==============================] - 502s 3s/step - loss: 0.1240 - acc: 0.9648 - val_loss: 0.5445 - val_acc: 0.8286
```
![](report_img/model_loss_35_0.png)
**说明：**
学习率提升为0.0003，增加衰减，尝试使用use_bias=False减少参数来防止过拟合
**36. 36**

**参数：**
- 模型: InceptionV3
- epochs = 30
- batch_size = 96
- 锁层: NO
- 停止提升轮数(patience): 3
- 自定义层:
  ```
  x = GlobalAveragePooling2D()(x)
  x = Dropout(0.5)(x)
  predictions = Dense(10, activation='softmax', use_bias=False, kernel_regularizer=l2(0.01))(x)
  ```
- 优化器: SGD
  - lr = 0.0003
  - decay = 30e-8

**结果：**
```

```

**说明：**
尝试对最后一层附加正则化来防止过拟合


在这一部分， 你需要描述你所建立的模型在给定数据上执行过程。模型的执行过程，以及过程中遇到的困难的描述应该清晰明了地记录和描述。需要考虑的问题：

你所用到的算法和技术执行的方式是否清晰记录了？
在运用上面所提及的技术及指标的执行过程中是否遇到了困难，是否需要作出改动来得到想要的结果？
是否有需要记录解释的代码片段(例如复杂的函数）？
### 完善
在这一部分，你需要描述你对原有的算法和技术完善的过程。例如调整模型的参数以达到更好的结果的过程应该有所记录。你需要记录最初和最终的模型，以及过程中有代表性意义的结果。你需要考虑的问题：

初始结果是否清晰记录了？
完善的过程是否清晰记录了，其中使用了什么技术？
完善过程中的结果以及最终结果是否清晰记录了？
## IV. 结果
（大概 2-3 页）

### 模型的评价与验证
在这一部分，你需要对你得出的最终模型的各种技术质量进行详尽的评价。最终模型是怎么得出来的，为什么它会被选为最佳需要清晰地描述。你也需要对模型和结果可靠性作出验证分析，譬如对输入数据或环境的一些操控是否会对结果产生影响（敏感性分析sensitivity analysis）。一些需要考虑的问题：

最终的模型是否合理，跟期待的结果是否一致？最后的各种参数是否合理？
模型是否对于这个问题是否足够稳健可靠？训练数据或输入的一些微小的改变是否会极大影响结果？（鲁棒性）
这个模型得出的结果是否可信？
### 合理性分析
在这个部分，你需要利用一些统计分析，把你的最终模型得到的结果与你的前面设定的基准模型进行对比。你也分析你的最终模型和结果是否确确实实解决了你在这个项目里设定的问题。你需要考虑：

最终结果对比你的基准模型表现得更好还是有所逊色？
你是否详尽地分析和讨论了最终结果？
最终结果是不是确确实实解决了问题？
## V. 项目结论
（大概 1-2 页）

### 结果可视化
在这一部分，你需要用可视化的方式展示项目中需要强调的重要技术特性。至于什么形式，你可以自由把握，但需要表达出一个关于这个项目重要的结论和特点，并对此作出讨论。一些需要考虑的：

你是否对一个与问题，数据集，输入数据，或结果相关的，重要的技术特性进行了可视化？
可视化结果是否详尽的分析讨论了？
绘图的坐标轴，标题，基准面是不是清晰定义了？
### 对项目的思考
在这一部分，你需要从头到尾总结一下整个问题的解决方案，讨论其中你认为有趣或困难的地方。从整体来反思一下整个项目，确保自己对整个流程是明确掌握的。需要考虑：

你是否详尽总结了项目的整个流程？
项目里有哪些比较有意思的地方？
项目里有哪些比较困难的地方？
最终模型和结果是否符合你对这个问题的期望？它可以在通用的场景下解决这些类型的问题吗？
### 需要作出的改进
在这一部分，你需要讨论你可以怎么样去完善你执行流程中的某一方面。例如考虑一下你的操作的方法是否可以进一步推广，泛化，有没有需要作出变更的地方。你并不需要确实作出这些改进，不过你应能够讨论这些改进可能对结果的影响，并与现有结果进行比较。一些需要考虑的问题：

是否可以有算法和技术层面的进一步的完善？
是否有一些你了解到，但是你还没能够实践的算法和技术？
如果将你最终模型作为新的基准，你认为还能有更好的解决方案吗？

## 参考文献
[1]Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. [
Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/pdf/1512.00567). arXiv:1512.00567, 2015.
[2]François Chollet. [Xception: Deep Learning with Depthwise Separable Convolutions](https://arxiv.org/pdf/1610.02357). arXiv prepr int arXiv:1610.02357, 2016.
[3]黄文坚. [CNN浅析和历年ImageNet冠军模型解析](http://www.infoq.com/cn/articles/cnn-and-imagenet-champion-model-analysis). 发表时间: 2017年5月22日.
[4]Kaggle. [State Farm Distracted Driver Detection](https://www.kaggle.com/c/state-farm-distracted-driver-detection). 2016.

**在提交之前， 问一下自己...**
- 你所写的项目报告结构对比于这个模板而言足够清晰了没有？
- 每一个部分（尤其分析和方法）是否清晰，简洁，明了？有没有存在歧义的术语和用语需要进一步说明的？
- 你的目标读者是不是能够明白你的分析，方法和结果？
- 报告里面是否有语法错误或拼写错误？
- 报告里提到的一些外部资料及来源是不是都正确引述或引用了？
- 代码可读性是否良好？必要的注释是否加上了？
- 代码是否可以顺利运行并重现跟报告相似的结果？
