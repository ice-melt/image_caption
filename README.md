# 看图说话机器人

项目安排：[原文地址点击这里][DOC01]&emsp;或&emsp;[当前项目中查看](项目安排.md)

项目简介：[原文地址点击这里][DOC02]&emsp;或&emsp;[当前项目中查看](项目简介.md)

## 相关资料参考

#### 1.Show and Tell

paper:&emsp;[Show and Tell: A Neural Image Caption Generator【arXiv:1411.4555】](arxiv14114555)

Blog:&emsp;[论文笔记：Show and Tell Lessons learned from the 2015 MSCOCO Image Captioning Challenge][blog001]

> 这篇博客对【arXiv:1411.4555】论文进行了翻译

Blog:&emsp;[Image caption——图像理解——看图说话综述（2015-2018）][blog002]

> 这篇博文给出了图像语义描述相关的一些方向

Blog:&emsp;[深度学习之image-caption系列（一）show and tell （NIC）模型理解与实现][blog003]

> 摘录了论文中的公式

code:&emsp;[https://gitee.com/littlemonky/program][code001]

> 参考代码

#### 2.Attention

attention 模型讲解:&emsp;[https://kexue.fm/archives/4765](https://kexue.fm/archives/4765)

attention 代码实现:&emsp;[https://github.com/bojone/attention](https://github.com/bojone/attention)

bottom-up-attention模型:&emsp;[https://github.com/peteanderson80/bottom-up-attention][code002]

> Github 上 top down or bottom up attention  的代码实现

neural baby talk 模型:  [https://github.com/jiasenlu/NeuralBabyTalk][code003]
> Github 上 新的 model 的实现方式 https://github.com/bojone/attention

#### 3.其他

[机器翻译自动评估-BLEU算法详解][blog004]
> 详细介绍了BLEU评价指标

## 数据集

[Flickr8k_Dataset][dataset_01]

[Flickr8k_text][dataset_02]

[Flickr8k TF-Record][dataset_03]

[Flickr30k TF-Record][dataset_04]

## 代码相关说明

#### 代码展示部分
- 修改 `configuration.py` 代码中相关代码
```python
class DisplayConfig(object):
  """Wrapper class for training hyperparameters."""

  def __init__(self):
    # self.checkpoint_path = "data/train_output/model.ckpt-1000"
    self.checkpoint_path = "data/output_mscoco/model.ckpt-200000"
    # self.vocab_file = "data/word_counts.txt"
    self.vocab_file = "data/word_counts_mscoco.txt"
```

> 此部分代码为新增代码，这里只需要修改模型的ckpt路径和词频表的路径即可

- 运行 `display.py`	


---

[DOC01]: https://gitee.com/ai100/projects-readme
[DOC02]: https://gitee.com/ai100/project-image-caption

[arxiv14114555]: https://arxiv.org/abs/1411.4555

[blog001]: https://blog.csdn.net/w5688414/article/details/79301976
[blog002]: https://blog.csdn.net/m0_37731749/article/details/80520144
[blog003]: https://blog.csdn.net/weixin_41694971/article/details/81359970
[blog004]: https://blog.csdn.net/qq_31584157/article/details/77709454

[code001]: https://gitee.com/littlemonky/program
[code002]: https://github.com/peteanderson80/bottom-up-attention
[code003]: https://github.com/jiasenlu/NeuralBabyTalk



[dataset_01]: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_Dataset.zip
[dataset_02]: http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/Flickr8k_text.zip
[dataset_03]: https://www.tinymind.com/fandichao1998/datasets/flickr8k
[dataset_04]: https://www.tinymind.com/fandichao1998/datasets/fk30kimagec

