from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from im2txt.ops import inputs as input_ops

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_integer("batch_size", "32",
                       "Batch size.")

tf.logging.set_verbosity(tf.logging.INFO)



def main(_):
    data_files = []
    data_files.extend(tf.gfile.Glob(FLAGS.input_file_pattern))
    print(data_files)

    if not data_files:
        tf.logging.fatal("Found no input files matching %s", FLAGS.input_file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), FLAGS.input_file_pattern)

    tfreader = tf.TFRecordReader()
    filename_queue = tf.train.string_input_producer(data_files, shuffle=True, capacity=16, name="filename_queue")
    values_queue = tf.RandomShuffleQueue(
        capacity=40000,
        min_after_dequeue=36800,
        dtypes=[tf.string],
        name="random_input_queue")
    _, value = tfreader.read(filename_queue)

    enqueue_ops = []
    enqueue_ops.append(values_queue.enqueue([value]))
    tf.train.queue_runner.add_queue_runner(tf.train.queue_runner.QueueRunner(values_queue, enqueue_ops))
    serialized_sequence_example = values_queue.dequeue()

    image_feature_name = "image/data"
    image_file_feature_name = "image/filename"
    # caption_feature_name = "image/caption_ids"

    # context, sequence = tf.parse_single_sequence_example(
    #     serialized_sequence_example,
    #     context_features={
    #         image_feature_name: tf.FixedLenFeature([], dtype=tf.string),
    #         image_file_feature_name: tf.FixedLenFeature([], dtype=tf.string)
    #     },
    #     sequence_features={
    #         caption_feature_name: tf.FixedLenSequenceFeature([], dtype=tf.int64)
    #     })
    context, sequence = tf.parse_single_sequence_example(
        serialized_sequence_example,
        context_features={
            image_feature_name: tf.FixedLenFeature([], dtype=tf.string),
            image_file_feature_name: tf.FixedLenFeature([], dtype=tf.string)
        })

    encoded_image = context[image_feature_name]
    image_name = context[image_file_feature_name]
    # caption = sequence[caption_feature_name]


    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ima_vals = []
        # cap_vals = []
        for i in range(1000):
            # ima_val, image_name_val, cap_val = sess.run([encoded_image, image_name, caption])
            ima_val, image_name_val = sess.run([encoded_image, image_name])

            # print('ima_val', ima_val)
            # print('cap_val', cap_val)
            # print('image_name_val', image_name_val)

            # ima_vals.append(ima_val)
            # cap_vals.append(cap_val)
        coord.request_stop()
        coord.join(threads)

if __name__ == "__main__":
    tf.app.run()