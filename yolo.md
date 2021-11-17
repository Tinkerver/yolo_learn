---
typora-copy-images-to: graph
typora-root-url: graph
---

# 目标检测算法-yolo系列

---

## 一、yolov1

### 1.实现方法

​	将图片分为s*s个网格，如果object中心落在网格中，则该网格负责检测此object。

​	网格需要预测b个bbox位置信息和置信度，位置信息四维置信度一维，confidence由iou和目标准确度相乘得到。每个网络还要预测长度为类别个数的向量，代表所得是哪一个类别。

​	在test时会把预测的类别信息和置信度相乘得到类别致信分数。这样得到的是带有类别的边界框，对它们设置阈值进行过滤最后得到最终的结果。

### 2.损失函数

​	损失函数使用了全平方误差，赋予坐标误差更大的损失权重，对置信度的框赋予小的损失权重。为了平衡大小box的影响，用width和height的平方根代替原本。

​	综上，在损失函数中，只有一个预测结果有object时才会对classification error 进行惩罚，通过类别预测的最大值去判断需要拟合哪一个标识框，并以此得到损失。

## 二、yolov2

### 1.简介

​	相对v1版本，提出了新的训练方法：联合训练。可以将两种数据集混合在一起使用分层的观点对物体分类。用分类数据集数据扩充检测数据集。联合训练算法希望同时在检测数据集和分类数据集上训练检测器。用检测数据集学习位置，用分类数据集增加分类类别量。

### 2.改进

​	通过使用batch normalization（批标准化）解决反向传播中梯度消失与梯度爆炸的问题，降低超参敏感性并起到正则化效果。

​	通过使用高分辨率图像分类，即对于原始224输入的参数提取网络权重通过448分辨率微调，来适应新分辨率的结果。

​	通过使用先验框，覆盖图像多种位置和尺度。这种方式让之前每个网格只预测两个变为了预测九个，预测出的框大大增多，因此去掉了全连接层。其中的物体更倾向于出现在中心位置，所以会有单独预测此类物体的位置。使用anchor box下降了精度，但可以让yolo预测的召回率上升。先验框的尺寸通过聚类的方式得到可以提高最后结果的map

​	通过约束预测边框的位置，将预测边框的中心约束在特定gird网格内

![image-20211115174958963](/image-20211115174958963.png)

​	

其中， ![[公式]](https://www.zhihu.com/equation?tex=b_x%2Cb_y%2Cb_w%2Cb_h) 是预测边框的中心和宽高。 ![[公式]](https://www.zhihu.com/equation?tex=Pr%28object%29%E2%88%97IOU%28b%2Cobject%29) 是预测边框的置信度，YOLO1是直接预测置信度的值，这里对预测参数 ![[公式]](https://www.zhihu.com/equation?tex=t_o) 进行σ变换后作为置信度的值。 ![[公式]](https://www.zhihu.com/equation?tex=c_x%2Cc_y) 是当前网格左上角到图像左上角的距离，要先将网格大小归一化，即令一个网格的宽=1，高=1。 ![[公式]](https://www.zhihu.com/equation?tex=p_w%2Cp_h) 是先验框的宽和高。 σ是sigmoid函数。 ![[公式]](https://www.zhihu.com/equation?tex=t_x%2Ct_y%2Ct_w%2Ct_h%2Ct_o) 是要学习的参数，分别用于预测边框的中心和宽高，以及置信度。

​	YOLO2引入一种称为passthrough层的方法在特征图中保留一些细节信息。具体来说，就是在最后一个pooling之前，特征图的大小是26*26*512，将其1拆4，直接传递（passthrough）到pooling后（并且又经过一组卷积）的特征图，两者叠加到一起作为输出的特征图。这样可以检测出一些比较小的对象。

​	为了将其应用在不同尺寸的图像上，yolov2在迭代时每几次迭代就会改变网络参数，十个batches就会随机选择一个新尺寸。这种方式可以在不同尺寸上都得到一个较好的结果。

## 三、yolov3

![img](https://pic3.zhimg.com/80/v2-5d97a1b944276ee2790febd230bb2112_720w.jpg)

### 1.结构	

​	DBL有darknetconv2d_BN_Leaky组成，有卷积+BN+Leaky组成。resn表示res_block有几个res_unit,concat时会对维度进行扩充。

​	骨干网络darknet-53，没有全连接层，因此可应对任意大小图像。用步长为2的卷积层替代之前的池化层。相较于resnet在精度相似的情况下速度提升。

​	yolov3中同样针对中心位置检测，格子负责中心坐标的检测。会有三个感受野，分别是32倍，16倍，8倍下采样。每个采样野中都会包括3个预选框。

​	坐标转换的设置与yolov2没有区别，对于未知的确定也是依靠预选框基础上的微调。

​	在先验框的选择中，是通过对训练样本进行k-means聚类得到。

​	用FPN的思路，设置多个探测头，多尺度预测。

​	使用了resnet结构，在residual-block中，更好的获取物体特征。

​	softmax用卷积层+logistic激活函数的结构替换。这样在class存在重合的情况下也能表现较好。

## 变种：tiny yolov3

​	通过减少尺度，只使用32倍下采样和16倍下采样预测来提升网络速度。

## 四、yolov4

### 1.结构

​	主要有backbone：cspdarknet53，neck：spp+pan，head：yolo head组成

#### backbone

​	是带有csp结构的darknet-53

![img](/v2-c7d4145838f7ffb009e0fb9bac00f0ed_720w.jpg)

​	CSPNet的主要目的是使该体系结构能够实现更丰富的梯度组合信息，同时减少计算量。 通过将基础层的特征图划分为两个部分，然后通过提出的跨阶段层次结构将它们合并，可以实现此目标。 我们的主要概念是通过分割梯度流，使梯度流通过不同的网络路径传播。 这样，我们已经确认，通过切换串联和过渡步骤，传播的梯度信息可以具有较大的相关性差异。

#### neck

​	neck部分主要用来融合不同尺寸的特征信息，应用spp+pan的方式。通过下采样拼接+上采样拼接来得到特征

#### yolo-head

​	检测头链接neck中所得到的特征分三个大小按不同格式输出。

### 2.损失函数

​	在使用均方误差时，中心坐标与宽高都被作为独立变量。但为了体现他们的关系，损失函数使用iou损失

#### iou损失

​	定义为1-iou（a，b）只有在重叠时才有效果，在未重叠时不会提供华东梯度。

#### giou损失

​	在原有iou损失上添加正则损失项![[公式]](https://www.zhihu.com/equation?tex=L_%7BGIOU%7D%3D1-IOU%28A%2CB%29%2B%5Cleft%7C+C-A%5Ccup+B%5Cright%7C%2F%5Cleft%7C+C%5Cright%7C)

![img](/v2-ed0a35eced7eb0dfa5e5aa429b83cc2c_720w.jpg)

可以在未重叠时表示损失。但会消耗大量的时间在预测框尝试与真实框接触上

#### diou损失

​	也添加了正则化项，正则化项为![[公式]](https://www.zhihu.com/equation?tex=L_%7BDIOU%7D%3D1-IOU%28A%2CB%29%2B%5Crho%5E%7B2%7D%28A_%7Bctr%7D%2CB_%7Bctr%7D%29%2Fc%5E%7B2%7D)

该惩罚项具体的参数含义为

- A : 预测框 B：真实框

- ![[公式]](https://www.zhihu.com/equation?tex=A_%7Bctr%7D) : 预测框**中心点坐标**

- ![[公式]](https://www.zhihu.com/equation?tex=B_%7Bctr%7D) ：真实框**中心点坐标**

- ![[公式]](https://www.zhihu.com/equation?tex=%5Crho%28.%29) 是**欧式距离**的计算

- c 为 A , B **最小包围框**的**对角线长度**

- ![img](/v2-d45b585bb6eea904a3779bc23864a1b2_720w.jpg)

#### ciou

​	**DIOU**考虑到了**两个检测框的中心距离**。而CIOU考虑到了三个**几何因素**，分别为重叠面积，中心点距离，长宽比。CIOU的公式定义如下![[公式]](https://www.zhihu.com/equation?tex=L_%7BCIOU%7D%3D1-IOU%28A%2CB%29%2B%5Crho%5E%7B2%7D%28A_%7Bctr%7D%2CB_%7Bctr%7D%29%2Fc%5E%7B2%7D%2B%5Calpha.v)

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha.v+) 对长宽比的惩罚项

![[公式]](https://www.zhihu.com/equation?tex=%5Calpha) 是一个正数， ![[公式]](https://www.zhihu.com/equation?tex=v) 用来测量长宽比的一致性（v measures the consistency of aspect ratio）。具体定义如下：

![img](/v2-d377d062705931befb0300d3ac69b89e_720w.jpg)

![img](/v2-bc194831f21dcb4d63597c93f1396410_720w.jpg)

- ![[公式]](https://www.zhihu.com/equation?tex=w%5E%7Bgt%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=h%5E%7Bgt%7D) 为真实框的宽、高
- ![[公式]](https://www.zhihu.com/equation?tex=w) 和 ![[公式]](https://www.zhihu.com/equation?tex=h) 为预测框的宽、高

​	若真实框和预测框的宽高相似，那么 ![[公式]](https://www.zhihu.com/equation?tex=v) 为0，该惩罚项就不起作用了。所以很直观地，这个惩罚项作用就是控制**预测框的宽高**能够尽可能**快速**地与**真实框的宽高**接近。

在损失函数方面，yolov4也就选用了ciou代替了yolov3的均方误差。



## 五、yolof

### 1.动机

​	对于FPN来讲，它采用了多尺度特征融合，提高了特征丰富度。并用分治法按照不同尺寸对不同子任务分别检测。

​	将检测器可抽象整在骨干网络后进行编解码两个操作。编码器处理backbone所提供的特征，解码器用于进行回归。

​	在实验中发现，多尺度特征融合带来的收益小于分治法所带来的收益。对目标检测按照大小分别拆分处理可能是FPN最大的优势。

### 2.改进点

#### dilated Encoder替换FPN

​	使用single in single out时感受野所对应的目标尺寸范围是受限的，无法应对剧烈变化的目标尺寸。使用空洞卷积则可以增加感受野，但对小目标的表达能力会变差。

​	dilated encoder可以解决这些问题：

![img](/v2-347bbbe090674ec22d7ca76b64fdacef_720w.jpg)

​	它将最底层c5特征作为输入，更改通道数目，3*3精炼语义，链接四个空洞卷积的残差单元，在空洞卷积中空洞的设定不一定相等。使用四个连续的空洞残差单元可以混合多种不同的感受野，也就应对不同的目标尺寸。

#### 解决正锚点不均匀

​	在之前的yolo网络中，很多是用iou》0.5来设为正锚点。在分而治之的机制下，不同尺寸都有相应锚点，也就存在充分数量的正锚点。但当使用single output时，锚点数量就会大量减少，也就导致了大的目标框包含更多正锚点，导致锚点不平衡。更容易忽视小目标。

​	为了解决这种不平衡，对每个目标框使用k近邻锚点作为正锚点，确保目标框能够以相同数量的正锚点匹配

### 3.网络结构

#### backbone

​	骨干网络使用resnet或resnetxt，输出2048\*32\*32或16\*16\*2048的特征图。这里从骨干网络获得的输出就都是单输出了，不再多层获取。

#### encoder

​	这里的encoder就是前面用的dilated encoder，变通道卷积+四个残差空洞单元组成的。

​	每个残差单元中，都会先用1*1减少通道到1/4，使用空洞卷积增大感受野之后再扩充回去。空洞卷积分别为2，4，6，8.

#### decoder

​	与retinanet的设置类似，回归分支包括四个Conv-BN-ReLU，分类包括两个Conv-BN-ReLU。

​	对于每个锚点框都有一个是否有目标置信度预测，最终分类输出要乘以目标置信度。
