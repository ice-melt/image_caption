# Show attend and tell 

[arXiv:1502.03044v3][https://arxiv.org/pdf/1502.03044v1.pdf]

视觉聚焦生成的看图说话神经网络



## 摘要

介绍了使用标准反向传播和随机最大化变化的下界，这类技术的一种确定性方式去训练我们这个模型。

通过可视化，我们展示了这个模型是如何聚焦在静态物体上，并且生成对应的输出的。



## 简介

模型任务：  1.确定什么物体在这幅图像中。2.捕捉他们之间的关系，并且以话的方式描述出来。

近些年来的工作成果： CNN 获取图像矢量表达特征，RNN解码这些特征称为自然语句



Attention： 相比较把整个图像压缩到静态表达中这种方法，attention 能够让显著特征的图像部分自动自动提前，对于图像嘈杂度很高的情况下特别有效。



作者观点： CNN低层的表示比高层的表示能够更好地保存住更丰富，具有描述性的表述。



介绍一个共同的框架下的attention 的两种变体：

**soft deterministic attention mechanism** :

通过传统的反向传播算法进行训练



**hard stochastic attention mechanism**:

最大化变化的下界或者增强学习的方法进行训练

 

Interpret the results of this framework by visualizing "where" and "what" the attention focused on.



![soft attend 和 hard attend的区别](pic\soft attend 和 hard attend的区别.jpg)

可以看出 soft attend 在聚焦一幅图中的某些特征的时候，面是比较散的。但是 hard attend 的话想对集中于莫一块，整个大小也相对一致 





## 相关工作

**Previous work:**

1. multimodal log-bilinear model that was biased by features from the image
2. explicity allow a natural way of doing both ranking and generation
3. replaced a feed-forward neural language model with a recurrent one.
4. LSTM RNNs 
5. see the image at each time step of the output word sequence
6. show the image to the RNN at the beginning
7. apply LSTMs to videos, allowing their model to generate video descriptions

之前的工作，都是把图片进行CNN卷积从最顶层抽出feature map 作为单个矢量特征。

8. learn a joint embedding space for ranking and generation whose model learns to score sentence and image similarity as a function of R-CNN object detections with outputs of a bidirectional RNN

9. a three-step pipline for generation by incorporating object detections.

   第一次提出基于多实例学习框架的，多种视觉概念探测器。



聚焦(attention)框架 不会进行显示地使用物体检测器，但是框架会从头学习他们潜在的匹配方式。

这样我们的模型就可以关注到一些抽象概念。



两种实现方式(fallen out of favour):

1. 通过物体识别和 属性探索去生成一个描述模板并进行相应词的填充
2. 从大量数据内获取一些相似描述的图像，修改他们去匹配现在的要求



这两种方式包含了中间的生成步骤，用于去除只是对获取的图片相关的描述的细节



## 模型

![模型主框架LSTM](pic\模型主框架LSTM.png)

模型相关解释请看 show and tell 算法理解 [https://blog.csdn.net/shenxiaolu1984/article/details/51493673]





**Modern work:**

First, two variants for our attention-based model by first describing their common framework.

 definition fo $\phi$ function

our model takes a single raw image and generates a caption y encoded as a sequence of 1-of-K encoded words.



[https://blog.csdn.net/shenxiaolu1984/article/details/51493673]: 