# BLEU机器翻译评价指标

BLEU(bilingual evaluation understudy)，**双语互译质量评估辅助工具**，主要用来评估机器翻译质量的工具。

**评判思想：**机器翻译结果越接近专业人工翻译的结果，则越好。

**实际工作：**判断两个**句子**的相似程度。

**计算公式：**
$$
BLEU-N=BP \cdot exp\Big(\sum_{n=1}^{N}{w_nlog{p_n}}\Big)
$$
其中，*BP*为惩罚因子，$p_n$为多元精度，$w_n$为多元精度对应的权重。

- [ ] ## 多元精度n-gram precision

- ### 原始多元精度

> 原文：猫坐在垫子上
> 机器译文： the the the the the the the.
> 参考译文：The cat is on the mat.

- 1元精度 1-gram

  ![img](https://upload-images.jianshu.io/upload_images/224008-594aba123f1c7e23?imageMogr2/auto-orient/strip%7CimageView2/2/w/400/format/webp)

  6个词中，5个词命中译文，1元精度$p_1$为5/6.

- 2元精度 2-gram

  ![img](https://upload-images.jianshu.io/upload_images/224008-81e46ed036c63c1b?imageMogr2/auto-orient/strip%7CimageView2/2/w/450/format/webp)

  2元词组的精度则是 3/5.

- 3元精度 3-gram

  ![img](https://upload-images.jianshu.io/upload_images/224008-6da346615eeef815?imageMogr2/auto-orient/strip%7CimageView2/2/w/450/format/webp)

  3元词组的精度为1/4.

- 4元精度 4-gram

  4元词组的精度为0。

一般情况，1-gram可以代表原文有多少词被单独翻译出来，可以反映译文的**充分性**，2-gram以上可以反映译文的**流畅性**，它的值越高说明**可读性**越好。

- 异常情况

  > 原文：猫坐在垫子上
  > 机器译文： the the the the the the the.
  > 参考译文：The cat is on the mat.

  此时，1-gram匹配度为7/7，显然，此译文翻译并不充分，此问题为**常用词干扰**。

- ### 改进多元精度

$$
Count^{clp}_{w_i,j}=min{(Count_{w_i},Ref_jCount_{w_i})}\\
Count^{clp}=max(Count^{clp}_{w_i,j}),i=1,2,3\cdots\\
p_n=\frac{\sum_{C\in{Candidates}}{\sum_{n-gram\in C}{Count_{clip}(n-gram)}}}{\sum_{C^{'}\in{Candidates}}{\sum_{n-gram^{'}\in C^{'}}{Count_{clip}(n-gram^{'})}}}
$$

其中，$Count_{w_i}$为单词$w_i$在机器译文中出现的次数，$Ref_jCount_{w_i}$为单词$w_i$在第$j$个译文中出现的次数，$Count^{clp}_{w_i,j}$为单词$w_i$对于第$j$个参考译文的截断计数，$Count^{clp}$为单词$w_i$在所有参考翻译里的综合截断计数，$p_n$为各阶N-gram的精度，$p_n$的公式分子部分表示$n$元组在翻译译文和各参考译文中出现的最小次数之和，分母部分表示$n$元组在各参考译文中出现的最大次数之和。

此时对于异常情况：$Count^{clp}=2$，此时，一元精度为2/7，避免了常用词干扰问题。

因此，改进的多元精度得分可以用来衡量翻译评估的**充分性**和**流畅性**两个指标。

- [ ] ## 多元精度组合

随着$n$的增大，精度得分总体成指数下降，采取几何加权平均，使各元精度起同等作用。

![è¿éåå¾çæè¿°](https://img-blog.csdn.net/20170830113001250?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvcXFfMzE1ODQxNTc=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)
$$
p_{ave}=exp\Big({\frac{1}{n}\cdot \sum_{i=1}^{N}{w_n\cdot log{p_n}}}\Big)
$$
其中，$p_{ave}$为多元精度组合值，$p_n$为n元精度，$w_n$为各元权重。

通常，BLEU-4为经典指标，$N$取4，$w_n$取1/4。

- [ ] ## 惩罚因子

$$
BP=
\begin{cases}
1\quad \quad \ \  if c>r\\
e^{1-{r}/{c}}\quad ifc\leq r
\end{cases}
$$

其中，$c$是机器译文的词数，$r$是参考译文的词数。

惩罚因子主要用来惩罚机器译文与参考译文长度差距过大情况。

- [ ] ## 总结

- 优点：

  方便、快速、结果有参考价值 

- 缺点：
  1. 不考虑语言表达（语法）上的准确性；
  2. 没有考虑同义词或相似表达的情况，可能会导致合理翻译被否定。



