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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import numpy as np
import rouge


import tensorflow as tf

from im2txt import configuration
from im2txt import inference_wrapper
from im2txt.inference_utils import caption_generator
from im2txt.inference_utils import vocabulary
from nltk.translate.bleu_score import sentence_bleu

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "", "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("input_file_token", "",
                       "Token input files.")
# CNN模型名称 InceptionV3 InceptionV4 DenseNet ResNet
tf.flags.DEFINE_string("CNN_name", "InceptionV3",
                       "CNN model name.")

tf.flags.DEFINE_string("input_file_names", "",
                       "输入图片文件名称列表.")

tf.logging.set_verbosity(tf.logging.INFO)

# 读取token
def load_token(images, token_path):
    id_to_captions = {}
    with tf.gfile.FastGFile(token_path, "r") as fc:
        lines = fc.readlines()
        for line in lines:
            ll = line.split("\t")
            filename = ll[0].split("#")[0]
            caption = ll[1]
            caption = caption.strip('\n')
            caption = caption.split(' ')
            if filename in images:
                id_to_captions.setdefault(filename, [])
                id_to_captions[filename].append(caption)

    return id_to_captions

# 计算BLEU指标
def sentences_tokens_BLEU(sentences, tokens, n_gram):
    # sentences:inference得到的句子列表
    # tokens:图片对应token列表
    # n_gram：n_gram BLEU
    score = 0

    for sentence in sentences:
        if n_gram == 1:
            score += sentence_bleu(tokens, sentence, weights=(1, 0, 0, 0))
        elif n_gram == 2:
            score += sentence_bleu(tokens, sentence, weights=(0.5, 0.5, 0, 0))
        elif n_gram == 3:
            score += sentence_bleu(tokens, sentence, weights=(0.33, 0.33, 0.33, 0))
        elif n_gram == 4:
            score += sentence_bleu(tokens, sentence, weights=(0.25, 0.25, 0.25, 0.25))
        else:
            tf.logging.error("BLEU n_gram： 1，2，3，4  (Error value %d)", n_gram)
    score /= len(tokens)

    return score

def sentences_tokens_BLEU_Max(sentences, tokens, n_gram):
    # sentences:inference得到的句子列表
    # tokens:图片对应token列表
    # n_gram：n_gram BLEU
    score = 0

    for sentence in sentences:
        if n_gram == 1:
            score = max(score, sentence_bleu(tokens, sentence, weights=(1, 0, 0, 0)))
        elif n_gram == 2:
            score = max(score, sentence_bleu(tokens, sentence, weights=(0.5, 0.5, 0, 0)))
        elif n_gram == 3:
            score = max(score, sentence_bleu(tokens, sentence, weights=(0.33, 0.33, 0.33, 0)))
        elif n_gram == 4:
            score = max(score, sentence_bleu(tokens, sentence, weights=(0.25, 0.25, 0.25, 0.25)))
        else:
            tf.logging.error("BLEU n_gram： 1，2，3，4  (Error value %d)", n_gram)

    return score

def prepare_results(p, r, f, metric):
    return '\t{}:\t{}: {:5.2f}\t{}: {:5.2f}\t{}: {:5.2f}'.format(metric, 'P', 100.0 * p, 'R', 100.0 * r, 'F1', 100.0 * f)

def sentences_tokens_Rouge(sentences=None, tokens=None, n_gram=4, metrics=None, aggregator='Best'):
    """
    计算Rouge指标
    sentences: inference结果caption句子列表
    tokens: 原图对应token列表
    n_gram: N-grams for ROUGE-N
    metrics: What ROUGE score to compute. Available: ROUGE-N, ROUGE-L, ROUGE-W. Default: ROUGE-N   可组合。['rouge-n', 'rouge-l', 'rouge-w']
    aggregator: 'Avg', 'Best', 'Individual'
    :return:
    """

    print('Rouge evaluation with {}'.format(aggregator))
    apply_avg = aggregator == 'Avg'
    apply_best = aggregator == 'Best'

    evaluator = rouge.Rouge(metrics=metrics,
                            max_n=n_gram,
                            limit_length=True,
                            length_limit=100,
                            length_limit_type='words',
                            apply_avg=apply_avg,
                            apply_best=apply_best,
                            alpha=0.5,  # Default F1_score
                            weight_factor=1.2,
                            stemming=True)

    for hypothesis in sentences:
        hypothesis = " ".join(hypothesis)
        for reference in tokens:
            reference = " ".join(reference)
            scores = evaluator.get_scores(hypothesis, reference)

    print_score = True
    if print_score:
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            if not apply_avg and not apply_best:  # value is a type of list as we evaluate each summary vs each reference
                for hypothesis_id, results_per_ref in enumerate(results):
                    nb_references = len(results_per_ref['p'])
                    for reference_id in range(nb_references):
                        print('\tHypothesis #{} & Reference #{}: '.format(hypothesis_id, reference_id))
                        print('\t' + prepare_results(results_per_ref['p'][reference_id],
                                                     results_per_ref['r'][reference_id],
                                                     results_per_ref['f'][reference_id]))
                print()
            else:
                print(prepare_results(results['p'], results['r'], results['f'], metric))
        print()

    return scores

def read_image_file_names():
    image_file_names = []

    if FLAGS.input_file_names:
        with tf.gfile.FastGFile(FLAGS.input_file_names, "r") as fc:
            lines = fc.readlines()
            for line in lines:
                image_file_names.append(line.strip('\n'))
    else:
        tf.logging.error("请指定参数:输入图片名称列表input_file_names！")

    print(image_file_names[:10])
    return image_file_names


def main(_):
    data_files = []
    data_files.extend(tf.gfile.Glob(FLAGS.input_file_pattern))
    print('input_file:', data_files)

    if not data_files:
        tf.logging.fatal("Found no input files matching %s", FLAGS.input_file_pattern)
    else:
        tf.logging.info("Prefetching values from %d files matching %s",
                        len(data_files), FLAGS.input_file_pattern)

    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper()
        model_config = configuration.ModelConfig()
        model_config.CNN_name = FLAGS.CNN_name
        restore_fn = model.build_graph_from_config(model_config,
                                                 FLAGS.checkpoint_path)


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

        context, sequence = tf.parse_single_sequence_example(
          serialized_sequence_example,
          context_features={
              image_feature_name: tf.FixedLenFeature([], dtype=tf.string),
              image_file_feature_name: tf.FixedLenFeature([], dtype=tf.string)
          })

        encoded_image = context[image_feature_name]
        image_name = context[image_file_feature_name]

        init = tf.global_variables_initializer()
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    total_images = read_image_file_names()
    tf.logging.info("Running caption generation on %d files matching %s",
                  len(total_images), FLAGS.input_file_pattern)

    image_to_tokens = load_token(total_images, FLAGS.input_file_token)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph=g) as sess:


        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # Load the model from checkpoint.
        restore_fn(sess)

        # Prepare the caption generator. Here we are implicitly using the default
        # beam search parameters. See caption_generator.py for a description of the
        # available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        # BLEU-1 BLEU-2 BLEU-3 BLEU-4
        BLEU = np.zeros(4)
        # Rouge precision, recall and f1 score
        Rouge = np.zeros(3)

        for i in range(3):

            ima_val, image_name_val = sess.run([encoded_image, image_name])

            image_file_name_val = str(image_name_val)
            image_file_name_val = image_file_name_val.split("\\")[-1]
            image_file_name_val = image_file_name_val[:-1]

            # 预测结果
            captions = generator.beam_search(sess, ima_val)
            caption_sentences = []
            print("Captions for image %s:" % os.path.basename(image_file_name_val))
            for i, caption in enumerate(captions):
                # Ignore begin and end words.
                sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
                caption_sentences.append(sentence)
                sentence = " ".join(sentence)
                print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))

            # 读取图片的token
            tokens = image_to_tokens[image_file_name_val]
            print("Tokens for image %s:" % image_file_name_val)
            for i, token in enumerate(tokens):
                print("  %d) %s" % (i, " ".join(token)))

            # 计算BLEU
            for i in range(4):
              # score = sentences_tokens_BLEU(caption_sentences, tokens, i + 1)
              score = sentences_tokens_BLEU_Max(caption_sentences, tokens, i + 1)
              BLEU[i] += score
              print("BLEU-%d for image %s: %f" % (i+1, image_file_name_val, score))

            # 计算Rouge （反应充分性和忠实性）
            Rouge_scores = sentences_tokens_Rouge(sentences=caption_sentences, tokens=tokens, n_gram=4, metrics=['rouge-l'])
            Rouge[0] += Rouge_scores['rouge-l']['p']
            Rouge[1] += Rouge_scores['rouge-l']['r']
            Rouge[2] += Rouge_scores['rouge-l']['f']

        image_num = len(total_images)
        print("Average BLEU-1 :", np.mean(BLEU[0]/image_num))
        print("Average BLEU-2 :", np.mean(BLEU[1]/image_num))
        print("Average BLEU-3 :", np.mean(BLEU[2]/image_num))
        print("Average BLEU-4 :", np.mean(BLEU[3]/image_num))

        print("Average ROUGE-L-precision :", np.mean(Rouge[0] / image_num))
        print("Average ROUGE-L-recall :", np.mean(Rouge[1] / image_num))
        print("Average ROUGE-L-F1 :", np.mean(Rouge[2] / image_num))


if __name__ == "__main__":
  tf.app.run()
