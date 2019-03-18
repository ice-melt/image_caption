# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Converts MSCOCO data to TFRecord file format with SequenceExample protos.

The MSCOCO images are expected to reside in JPEG files located in the following
directory structure:

  train_image_dir/COCO_train2014_000000000151.jpg
  train_image_dir/COCO_train2014_000000000260.jpg
  ...

and

  val_image_dir/COCO_val2014_000000000042.jpg
  val_image_dir/COCO_val2014_000000000073.jpg
  ...

The MSCOCO annotations JSON files are expected to reside in train_captions_file
and val_captions_file respectively.

This script converts the combined MSCOCO data into sharded data files consisting
of 256, 4 and 8 TFRecord files, respectively:

  output_dir/train-00000-of-00256
  output_dir/train-00001-of-00256
  ...
  output_dir/train-00255-of-00256

and

  output_dir/val-00000-of-00004
  ...
  output_dir/val-00003-of-00004

and

  output_dir/test-00000-of-00008
  ...
  output_dir/test-00007-of-00008

Each TFRecord file contains ~2300 records. Each record within the TFRecord file
is a serialized SequenceExample proto consisting of precisely one image-caption
pair. Note that each image has multiple captions (usually 5) and therefore each
image is replicated multiple times in the TFRecord files.

The SequenceExample proto contains the following fields:

  context:
    image/image_id: integer MSCOCO image identifier
    image/data: string containing JPEG encoded image in RGB colorspace

  feature_lists:
    image/caption: list of strings containing the (tokenized) caption words
    image/caption_ids: list of integer ids corresponding to the caption words

The captions are tokenized using the NLTK (http://www.nltk.org/) word tokenizer.
The vocabulary of word identifiers is constructed from the sorted list (by
descending frequency) of word tokens in the training set. Only tokens appearing
at least 4 times are considered; all other words get the "unknown" word id.

NOTE: This script will consume around 100GB of disk space because each image
in the MSCOCO dataset is replicated ~5 times (once per caption) in the output.
This is done for two reasons:
  1. In order to better shuffle the training data.
  2. It makes it easier to perform asynchronous preprocessing of each image in
     TensorFlow.

Running this script using 16 threads may take around 1 hour on a HP Z420.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function #获取python未来版本的功能，解决兼容性问题

from collections import Counter #对字符串中各字符计数，生成{字符:数目}的字典
from collections import namedtuple #命名元组，实例化后相当于字典，name.items为[("id",1)]
from datetime import datetime #表示日期时间
import json #一种文件读写模块
import os.path  #操作系统路径
import random #随机数
import sys #程序与python解释器交互 os:程序与操作系统交互
import threading #线程模块
import os


import nltk.tokenize #句子标注
import numpy as np
from six.moves import xrange #six:处理2,3,兼容性
import tensorflow as tf

#需要下载nltk 插件中的punkt 如果没用下载第一次需要使用
# nltk.download('punkt')

##################################################################输入定义#########################################################
#数据集名称 mscoco, flickr30k,flickr8k
tf.flags.DEFINE_string("dataset_name", r"mscoco","name of dataset ,must be flickr30k or flickr8k or mscoco.")

#mscoco  训练图片文件夹
#flickr8k, flickr30k train 和 val 图片文件夹
tf.flags.DEFINE_string("train_image_dir", r"D:\MSCOCODATA\trainData","Training image file.")

#mscoco  验证图片文件夹
tf.flags.DEFINE_string("val_image_dir", r"D:\MSCOCODATA\valData","Validation image file.")

#flickr8k  包含train 图片数据集文件名称的文件， flickr8k  Flickr_8k.trainImages.txt
#flickr30k train 和 val 数据集地址，只是文件夹
tf.flags.DEFINE_string("train_file_path", r"D:\MSCOCODATA\anno\captions_train2014.json","Training image file.")

#flickr8k  包含val 图片数据集文件名称的文件, flickr8k  Flickr_8k.devImages.txt
tf.flags.DEFINE_string("val_file_path", r"D:\MSCOCODATA\anno\captions_val2014.json","Validation image file.")

# flickr8k flickr30k 的训练、验证数据集的 Caption 文件  Flickr8k.token.txt， mscoco 的训练集 Caption File 位置  mscoco: captions_train2014.json
tf.flags.DEFINE_string("train_captions_file", r"D:\MSCOCODATA\anno\captions_train2014.json","Testing image file.")

# mscoco 的验证集 Caption 文件 mscoco  captions_val2014.json
tf.flags.DEFINE_string("val_captions_file", r"D:\MSCOCODATA\anno\captions_val2014.json","Validation captions JSON file.")
##################################################################输出定义#########################################################

# 输出测试集，训练集数据集文件名称地址
tf.flags.DEFINE_string("output_filename_path", r"E:\imageCaptionDataset\mscoco\filename", "Image file names' txt")

#输出图片位置
tf.flags.DEFINE_string("output_dir", r"E:\imageCaptionDataset\mscoco\dataset", "Output data directory.")

# 词频字典生成地址 自动生成名称为 word_counts.txt的文件，没有文件夹会自动创建文件
tf.flags.DEFINE_string("word_counts_output_file", r"E:\imageCaptionDataset\mscoco\word",
                       "Output vocabulary file of word counts.")

##################################################################参数定义#########################################################
#只有flickr_30k 有效 ,定义数据集中定义中的训练数据集的数量
tf.flags.DEFINE_integer("nums_for_train", 28000,
                        "Number of train images in training TFRecord files.")


#训练数据分区，Mscoco 数据集分为 256 ，flickr_30k 分为8，flickr_8k分为4
tf.flags.DEFINE_integer("train_shards", 256,
                        "Number of shards in training TFRecord files.")
#验证集数据集分区，Mscoco 数据集分为 4 ,flickr_30k 和 flickr_8k 分为1
tf.flags.DEFINE_integer("val_shards", 4,
                        "Number of shards in validation TFRecord files.")

# Caption 开始符号自定义
tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")

# Caption 结束符号自定义
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
# Caption 词频小于了 min_word_count 的数量所有的词对应的词频表中字符
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
# 最小词频数量数定义
tf.flags.DEFINE_integer("min_word_count", 4,
                        "The minimum number of occurrences of each word in the "
                        "training set for inclusion in the vocabulary.")

# 制作tf-record的线程数量
tf.flags.DEFINE_integer("num_threads", 8,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS #处理命令行参数，后续可通过命令行向程序中传递参数

ImageMetadata = namedtuple("ImageMetadata",
                           ["image_id", "filename", "captions"])#命名元组 ImageMetadata({image_id:123344})


class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self, vocab, unk_id):
    """Initializes the vocabulary.

    Args:
      vocab: A dictionary of word to word_id.
      unk_id: Id of the special 'unknown' word.
    """
    self._vocab = vocab
    self._unk_id = unk_id

  def word_to_id(self, word):
    """Returns the integer id of a word string."""
    if word in self._vocab:
      return self._vocab[word]
    else:
      return self._unk_id


class ImageDecoder(object):
  """Helper class for decoding images in TensorFlow."""

  def __init__(self):
    # Create a single TensorFlow Session for all image decoding calls.
    self._sess = tf.Session()

    # TensorFlow ops for JPEG decoding.
    self._encoded_jpeg = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._encoded_jpeg, channels=3)

  def decode_jpeg(self, encoded_jpeg):
    image = self._sess.run(self._decode_jpeg,
                           feed_dict={self._encoded_jpeg: encoded_jpeg})
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image


def _int64_feature(value):
  """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value.encode()if type(value)==str else value]))


def _int64_feature_list(values):
  """Wrapper for inserting an int64 FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _bytes_feature_list(values):
  """Wrapper for inserting a bytes FeatureList into a SequenceExample proto."""
  return tf.train.FeatureList(feature=[_bytes_feature(v) for v in values])


def _to_sequence_example(image, decoder, vocab):
  """Builds a SequenceExample proto for an image-caption pair.

  Args:
    image: An ImageMetadata object.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.

  Returns:
    A SequenceExample proto.
  """
  with tf.gfile.FastGFile(image.filename, "rb") as f:
    encoded_image = f.read()

  try:
    decoder.decode_jpeg(encoded_image)
  except (tf.errors.InvalidArgumentError, AssertionError):
    print("Skipping file with invalid JPEG data: %s" % image.filename)
    return

  image_name = ""
  if os.path.split(image.filename):
    image_name = os.path.split(image.filename)[-1]

  # print(image_name)
  context = tf.train.Features(feature={
      "image/image_id": _int64_feature(image.image_id),
      "image/data": _bytes_feature(encoded_image),
      "image/filename":_bytes_feature(image_name)
  })

  assert len(image.captions) == 1
  caption = image.captions[0]
  caption_ids = [vocab.word_to_id(word) for word in caption]
  feature_lists = tf.train.FeatureLists(feature_list={
      "image/caption": _bytes_feature_list(caption),
      "image/caption_ids": _int64_feature_list(caption_ids)
  })
  sequence_example = tf.train.SequenceExample(
      context=context, feature_lists=feature_lists)

  return sequence_example


def _process_image_files(thread_index, ranges, name, images, decoder, vocab,
                         num_shards):
  """Processes and saves a subset of images as TFRecord files in one thread.

  Args:
    thread_index: Integer thread identifier within [0, len(ranges)].
    ranges: A list of pairs of integers specifying the ranges of the dataset to
      process in parallel.
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    decoder: An ImageDecoder object.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Each thread produces N shards where N = num_shards / num_threads. For
  # instance, if num_shards = 128, and num_threads = 2, then the first thread
  # would produce shards [0, 64).
  num_threads = len(ranges)
  assert not num_shards % num_threads
  num_shards_per_batch = int(num_shards / num_threads)

  shard_ranges = np.linspace(ranges[thread_index][0], ranges[thread_index][1],
                             num_shards_per_batch + 1).astype(int)
  num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

  counter = 0
  for s in xrange(num_shards_per_batch):#32
    # Generate a sharded version of the file name, e.g. 'train-00002-of-00010'
    shard = thread_index * num_shards_per_batch + s
    output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
    output_file = os.path.join(FLAGS.output_dir, output_filename) #连接目录与文件名或目录
    writer = tf.python_io.TFRecordWriter(output_file)

    shard_counter = 0
    images_in_shard = np.arange(shard_ranges[s], shard_ranges[s + 1], dtype=int)
    for i in images_in_shard: #[0,1,2]...[123,124]...共32
      image = images[i]

      sequence_example = _to_sequence_example(image, decoder, vocab)
      if sequence_example is not None:
        writer.write(sequence_example.SerializeToString())
        shard_counter += 1
        counter += 1

      if not counter % 1000:
        print("%s [thread %d]: Processed %d of %d items in thread batch." %
              (datetime.now(), thread_index, counter, num_images_in_thread))
        sys.stdout.flush()

    writer.close()
    print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
          (datetime.now(), thread_index, shard_counter, output_file))
    sys.stdout.flush()#刷新输出
    shard_counter = 0
  print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
        (datetime.now(), thread_index, counter, num_shards_per_batch))
  sys.stdout.flush()


def _process_dataset(name, images, vocab, num_shards):
  """Processes a complete data set and saves it as a TFRecord.

  Args:
    name: Unique identifier specifying the dataset.
    images: List of ImageMetadata.
    vocab: A Vocabulary object.
    num_shards: Integer number of shards for the output files.
  """
  # Break up each image into a separate entity for each caption.
  images = [ImageMetadata(image.image_id,image.filename, [caption])
            for image in images for caption in image.captions]

  # Shuffle the ordering of images. Make the randomization repeatable.
  random.seed(12345) #使多次生成的随机数相同
  random.shuffle(images) #打乱图片顺序

  # Break the images into num_threads batches. Batch i is defined as
  # images[ranges[i][0]:ranges[i][1]].
  num_threads = min(num_shards, FLAGS.num_threads)
  spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
  ranges = []
  threads = []
  for i in xrange(len(spacing) - 1):
    ranges.append([spacing[i], spacing[i + 1]]) #[0,3750],[3750,7500]。。。。

  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator() 

  # Create a utility for decoding JPEG images to run sanity checks.
  decoder = ImageDecoder()

  # Launch a thread for each batch.
  print("Launching %d threads for spacings: %s" % (num_threads, ranges))
  for thread_index in xrange(len(ranges)):  #多线程处理的重点
    args = (thread_index, ranges, name, images, decoder, vocab, num_shards)
    t = threading.Thread(target=_process_image_files, args=args)#创建一个线程
    t.start()#启动线程
    threads.append(t)

  # Wait for all the threads to terminate.
  coord.join(threads)
  print("%s: Finished processing all %d image-caption pairs in data set '%s'." %
        (datetime.now(), len(images), name))


def _create_vocab(captions):
  """Creates the vocabulary of word to word_id.

  The vocabulary is saved to disk in a text file of word counts. The id of each
  word in the file is its corresponding 0-based line number.

  Args:
    captions: A list of lists of strings.

  Returns:
    A Vocabulary object.
  """
  print("Creating vocabulary.")
  counter = Counter()
  for c in captions:
    counter.update(c) #对列表中各字符计数，并更新
  print("Total words:", len(counter))

  # 降序排词，如果小于最小词频数，记为unk,词频表中不进行显示.
  word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
  word_counts.sort(key=lambda x: x[1], reverse=True)
  print("Words in vocabulary:", len(word_counts))

  #判断是否存在word_count的输出文件夹，没有进行添加
  if not tf.gfile.IsDirectory(FLAGS.word_counts_output_file):
    tf.gfile.MakeDirs(FLAGS.word_counts_output_file)

  wc_path = os.path.join(FLAGS.word_counts_output_file,"word_counts.txt")
  print("Wrote vocabulary file:", wc_path)

  # Write out the word counts file.
  with tf.gfile.FastGFile(wc_path, "w") as f:
    f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))



  # 创建词频表。并且在Vocabulary中用 词频表排列的序列对
  reverse_vocab = [x[0] for x in word_counts]
  unk_id = len(reverse_vocab)
  vocab_dict = dict([(x, y) for (y, x) in enumerate(reverse_vocab)])
  vocab = Vocabulary(vocab_dict, unk_id)

  return vocab


def _process_caption(caption):
  """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.

  Returns:
    A list of strings; the tokenized caption.
  """

  tokenized_caption = [FLAGS.start_word]
  tokenized_caption.extend(nltk.tokenize.word_tokenize(caption.lower()))#将句子拆分成单词列表并将其添加到token...
  tokenized_caption.append(FLAGS.end_word)
  return tokenized_caption


def _load_and_process_metadata(captions_file,image_dir,dataset_name,filenames=None):
  """Loads image metadata from a JSON file and processes the captions.

  Args:
    captions_file: JSON file containing caption annotations.
    image_dir: Directory containing the image files.

  Returns:
    A list of ImageMetadata.
  """
  id_to_captions = {}

  #根据caption 获取imageid_to_caption 字典
  if dataset_name == "flickr8k" or dataset_name == "flickr30k":
    filenames_str = ",".join(filenames)
    print("length of filename:", len(filenames))

    with tf.gfile.FastGFile(captions_file, "r") as fc:
      lines = fc.readlines()
      for line in lines:
        ll = line.split("\t")
        filename = ll[0].split("#")[0]
        caption = ll[1]
        if filenames_str.find(filename) != -1:
          id_to_captions.setdefault(filename, [])
          id_to_captions[filename].append(caption)




  elif dataset_name == "mscoco":
    with tf.gfile.FastGFile(captions_file, "r") as f:
      caption_data = json.load(f)
    # Extract the filenames.
      filenames = [(x["id"], x["file_name"]) for x in caption_data["images"]]
    # Extract the captions. Each image_id is associated with multiple captions.
    for annotation in caption_data["annotations"]:
      image_id = annotation["image_id"]
      caption = annotation["caption"]
      id_to_captions.setdefault(image_id, [])
      id_to_captions[image_id].append(caption)

  print("length of id_to_captions:",len(id_to_captions))
  print("length of filename",len(filenames))
  assert len(filenames) == len(id_to_captions)
  if dataset_name == "flickr8k" or dataset_name == "flickr30k":
    assert set([x for x in filenames]) == set(id_to_captions.keys())
  elif dataset_name == "mscoco":
    assert set([x[0] for x in filenames]) == set(id_to_captions.keys())

  print("Loaded caption metadata for %d images from %s" %
         (len(filenames), captions_file))

  # Process the captions and combine the data into a list of ImageMetadata.
  print("Processing captions.")
  image_metadata = []
  num_captions = 0
  filename_iter = None
  if dataset_name == "flickr8k" or dataset_name == "flickr30k":
    for image_id,base_filename in enumerate(filenames):
      filename = os.path.join(image_dir, base_filename)
      captions = [_process_caption(c) for c in id_to_captions[base_filename]]
      image_metadata.append(ImageMetadata(image_id+1,filename, captions))
      num_captions += len(captions)

  elif dataset_name == "mscoco":
    for image_id, base_filename in filenames:
      filename = os.path.join(image_dir, base_filename)
      captions = [_process_caption(c) for c in id_to_captions[image_id]]
      image_metadata.append(ImageMetadata(image_id, filename, captions))
      num_captions += len(captions)

  print("Finished processing %d captions for %d images in %s" %(num_captions, len(filenames), captions_file))

  return image_metadata

def _read_filename(image_file):
  with tf.gfile.FastGFile(image_file,"r") as ff: #对图片文件的读取
      filenames_l = ff.readlines()
      filenames = []
      for i in filenames_l:
        filenames.append(i.strip())
  return filenames

def _read_coco_filename(image_file):
  with tf.gfile.FastGFile(image_file, "r") as f:
    jsondata = json.load(f)
    filenames = [x["file_name"] for x in jsondata["images"]]
  return filenames

def _read_files_and_split_data(dataset_name,train_image_dir,test_image_dir):
  train_images,val_images = [],[]

  if dataset_name == "flickr30k":
    total_images = os.listdir(train_image_dir)
    random.seed(7654321)
    random.shuffle(total_images)
    train_images = total_images[:FLAGS.nums_for_train]
    val_images = total_images[FLAGS.nums_for_train:]

  elif dataset_name == "flickr8k":
    train_images = _read_filename(train_image_dir)
    val_images = _read_filename(test_image_dir)
  elif dataset_name == "mscoco":
    train_images = _read_coco_filename(train_image_dir)
    val_images = _read_coco_filename(test_image_dir)

  return train_images,val_images

def _output_filename_process(train_filenames,val_filenames):
  file_path = FLAGS.output_filename_path
  train_images_dir = os.path.join(file_path, "train_images_dir.txt")
  val_images_dir = os.path.join(file_path, "val_images_dir.txt")

  with open(train_images_dir, "w") as f1:
    for line in train_filenames:
      f1.write(line + '\n')

  with open(val_images_dir, "w") as f2:
    for line1 in val_filenames:
      f2.write(line1 + '\n')



def _moscoco_train_val_build(train,val):
  train_cutoff = int(0.85 * len(val))
  val_cutoff = int(0.90 * len(val))
  modified_train = train + val[0:train_cutoff]
  print("mscoco total train image numbers:",len(train)+train_cutoff)
  print("mscoco total val image numbers:",val_cutoff-train_cutoff)
  modified_val = train[train_cutoff:val_cutoff]
  return modified_train,modified_val


def main(unused_argv):
  dataset_name = FLAGS.dataset_name

  #检查分区数量能否被线程正确操作
  def _is_valid_num_shards(num_shards):
    """Returns True if num_shards is compatible with FLAGS.num_threads."""
    return num_shards < FLAGS.num_threads or not num_shards % FLAGS.num_threads

  assert _is_valid_num_shards(FLAGS.train_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.train_shards")
  assert _is_valid_num_shards(FLAGS.val_shards), (
      "Please make the FLAGS.num_threads commensurate with FLAGS.val_shards")



  #判断输出目录是否
  if not tf.gfile.IsDirectory(FLAGS.output_dir): #判断所给目录是否存在
    tf.gfile.MakeDirs(FLAGS.output_dir) #创建目录



  raw_train_dataset = []
  raw_val_dataset = []
  train_filename = []
  val_filename = []
  train_dataset = []
  val_dataset = []

  # 根据Caption 生成 ImageMetaData 的List
  if dataset_name == "mscoco":
    train_filename, val_filename = _read_files_and_split_data(dataset_name, FLAGS.train_captions_file, FLAGS.val_captions_file)
    raw_train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,FLAGS.train_image_dir,dataset_name)
    raw_val_dataset = _load_and_process_metadata(FLAGS.val_captions_file,FLAGS.val_image_dir,dataset_name)

  elif dataset_name == "flickr30k" or dataset_name == "flickr8k":
    train_filename, val_filename = _read_files_and_split_data(dataset_name,FLAGS.train_file_path,FLAGS.val_file_path)
    raw_train_dataset = _load_and_process_metadata(FLAGS.train_captions_file,FLAGS.train_image_dir,dataset_name,train_filename)
    raw_val_dataset = _load_and_process_metadata(FLAGS.train_captions_file,FLAGS.train_image_dir,dataset_name,val_filename)


  #根据文件名的不同切分训练集和测试集
  modified_train_filename = ""
  modified_val_filename = ""

  if dataset_name == "mscoco":
    # Redistribute the MSCOCO data as follows:
    #   train_dataset = 100% of mscoco_train_dataset + 85% of mscoco_val_dataset.
    #   val_dataset = 5% of mscoco_val_dataset (for validation during training).
    #   test_dataset = 10% of mscoco_val_dataset (for final evaluation).
    train_dataset,val_dataset = _moscoco_train_val_build(raw_train_dataset,raw_val_dataset)
    modified_train_filename,modified_val_filename = _moscoco_train_val_build(train_filename,val_filename)

  elif dataset_name == "flickr30k" or dataset_name == "flickr8k":
    train_dataset,val_dataset = raw_train_dataset,raw_val_dataset
    modified_train_filename, modified_val_filename = train_filename,val_filename

  #输出训练集文件名，验证集文件名
  _output_filename_process(modified_train_filename,modified_val_filename)

  # 制作Caption
  train_captions = [c for image in train_dataset for c in image.captions]
  vocab = _create_vocab(train_captions) #返回一个词汇处理的函数

  #制作tf record
  _process_dataset("train", train_dataset, vocab, FLAGS.train_shards)
  _process_dataset("val", val_dataset, vocab, FLAGS.val_shards)


if __name__ == "__main__":
  tf.app.run()#解析命令行后，执行main()函数
