# Image Caption踩到的坑

## 1. tinymind 上 布尔值变量为'true'，python里需要'True',截图如下：

![PIC01][PIC01]

> 参数显示 小写'true'

![PIC02][PIC02]

> 提示 train_CNN 需要一个参数

#### 解决办法：

1. 可以修改代码参数默认值为'True',然后取消tinymind上的此参数

```python
	parser.add_argument('--train_CNN', type=bool, default=True)
```

2. 避免使用tinymind上的布尔值变量，该用int类型变量(0和1),代码中根据0和1进行三目判断


## 2. tinymind 运行过程中内存不足

![PIC03][PIC03]

#### 解决办法：

1. 这里是将`batch_size`从16降到了4


[PIC01]:https://raw.github.com/CSDN-AI7/image_caption/master/Documents/resources/xxliu_pic01.png
[PIC02]:https://raw.github.com/CSDN-AI7/image_caption/master/Documents/resources/xxliu_pic02.png
[PIC03]:https://raw.github.com/CSDN-AI7/image_caption/master/Documents/resources/xxliu_pic03.png