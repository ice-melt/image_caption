# 遇到的问题

## 1.  tfrecord 图片文件读取方式不对

```python
这里读取的 r要改为rb
with tf.gfile.FastGFile(image.filename, "rb") as f:
  encoded_image = f.read()
```



否则会出现

UnicodeDecodeError: 'utf-8' codec can't decode byte 0xff in position 0: invalid start byte

报错

## 2.  tfrecord 制作时的 windows 下编码格式问题

linux 下面不会出现 windows 下面会报错

tensorflow.python.framework.errors_impl.NotFoundError: NewRandomAccessFile failed to Create/Open: E:\flick\data\Flickr_8k.trainImages.txt : ϵͳ\udcd5Ҳ\udcbb\udcb5\udcbdָ\udcb6\udca8\udcb5\udcc4·\udcbe\udcb6\udca1\udca3
; No such process



这段代码：

```python
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
```

需要对  value 进行编码 而不是直接 str 应该改成：



```python
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()if type(value)==str else value]))
```



这里要根据值得类型进行判断 如果不是 str的类型要进行encode 编码



## 3. 梯度爆炸问题

你先查查梯度爆炸的原因 ： 有可能是优化器问题， 有可能是学习率设置问题

调参时候 学习率设置问题 和 优化器选择问题出错导致 loss直线上升

![梯度爆炸1](pic\梯度爆炸1.jpg)



## 4. 重点问题，原来代码的问题

制作tf record 速度问题解决 

在制作 tfrecord 的 namedTuple 时，读取 caption_file 中 会使用一句话 

```python
with tf.gfile.FastGFile(captions_file, "r") as fc:
  lines = fc.readlines()
  id_to_captions = {}
  for line in lines:
    ll = line.split("\t")
    filename = ll[0].split("#")[0]
    caption = ll[1]
    if filename in filenames:
      id_to_captions.setdefault(filename, [])
      id_to_captions[filename].append(caption)
```



用来把  每个图片对应的语句给放到 id_to_captions 的字典中

filenames 是一个列表

if filename in filenames 速度非常慢，而且外循环 for line in lines 造成速度慢的原因。

把filenames 这个列表 转化为 filenames_str 的 字符串 使用逗号隔开文件名， 然后使用 

if  filenames.find(filename) != -1  去 替换掉 if filename in filenames  速度将会大大加快，这在制作flickr30k 的时候尝试过。

## 5. GPU 配置不对的问题训练facenet中的train_softmax.py时，出现了以下错误

failed to create cublas handle: CUBLAS_STATUS_ALLOC_FAILED

这个是由于GPU的配置不对造成的。

解决办法：

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config, ...)
