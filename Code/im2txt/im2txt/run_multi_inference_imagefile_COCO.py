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
import json


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
tf.flags.DEFINE_string("input_file_token", "",
                       "Token input files.")
tf.flags.DEFINE_string("input_file_image", "",
                       "输入图片文件夹路径（测试版）.")
# CNN模型名称 InceptionV3 InceptionV4 DenseNet ResNet
tf.flags.DEFINE_string("CNN_name", "InceptionV3",
                       "CNN model name.")

tf.logging.set_verbosity(tf.logging.INFO)

# 读取token json文件
def load_token(token_path):
    id_to_image = dict()
    id_to_annotation = dict()
    image_to_caption = dict()

    with open(token_path, 'r') as f:
        line = f.readline()
        load_data = json.loads(line)
        image_data = load_data['images']
        annotation_data = load_data['annotations']
        for image_info in image_data:
            image_id = image_info['id']
            id_to_image.setdefault(image_id, [])
            id_to_image[image_id].append(image_info['file_name'])

        for annotation_info in annotation_data:
            image_id = annotation_info['image_id']
            id_to_annotation.setdefault(image_id, [])
            id_to_annotation[image_id].append(annotation_info['caption'])

        for i in id_to_image:
            file_name = "".join(id_to_image[i])
            annotation = list(id_to_annotation[i])

            image_to_caption.setdefault(file_name, [])

            for i in range(len(annotation)):
                ann = annotation[i].strip('.').split(' ')
                ann.append('.')
                # print('ann ', ann)
                image_to_caption[file_name].append(ann)


        del load_data
        del image_data
        del annotation_data
        del id_to_image
        del id_to_annotation

        f.close()

    return image_to_caption

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

def main(_):
  # Build the inference graph.
  g = tf.Graph()
  with g.as_default():
      model = inference_wrapper.InferenceWrapper()
      model_config = configuration.ModelConfig()
      model_config.CNN_name = FLAGS.CNN_name
      restore_fn = model.build_graph_from_config(model_config,
                                                 FLAGS.checkpoint_path)
  g.finalize()

  # Create the vocabulary.
  vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

  image_to_tokens = load_token(FLAGS.input_file_token)

  with tf.Session(graph=g) as sess:

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

      image_num = len(image_to_tokens)
      cur_num = 0

      for image_name in image_to_tokens:
          filename = os.path.join(FLAGS.input_file_image, image_name)
          if not os.path.exists(filename):
              # print('文件不存在', filename)
              continue
          with tf.gfile.GFile(filename, "rb") as f:
              image = f.read()
          # 预测结果
          captions = generator.beam_search(sess, image)
          caption_sentences = []
          print("Captions for image %s:" % os.path.basename(filename))
          for i, caption in enumerate(captions):
              # Ignore begin and end words.
              sentence = [vocab.id_to_word(w) for w in caption.sentence[1:-1]]
              caption_sentences.append(sentence)
              sentence = " ".join(sentence)
              print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
              break

          # 读取图片的token
          tokens = image_to_tokens[image_name]
          # print("Tokens for image %s:" % image_name)
          # for i, token in enumerate(tokens):
          #     print("  %d) %s" % (i, " ".join(token)))

          # print('caption_sentences', caption_sentences)
          # print('tokens', tokens)

          # 计算BLEU
          for i in range(4):
              # score = sentences_tokens_BLEU(caption_sentences, tokens, i + 1)
              score = sentences_tokens_BLEU_Max(caption_sentences, tokens, i + 1)
              BLEU[i] += score
              print("BLEU-%d for image %s: %f" % (i+1, image_name, score))

          # 计算Rouge （反应充分性和忠实性）
          Rouge_scores = sentences_tokens_Rouge(sentences=caption_sentences, tokens=tokens, n_gram=4, metrics=['rouge-l'])
          Rouge[0] += Rouge_scores['rouge-l']['p']
          Rouge[1] += Rouge_scores['rouge-l']['r']
          Rouge[2] += Rouge_scores['rouge-l']['f']

          cur_num += 1
          if cur_num % 10 == 0 :
            print("Inference progress rate:", cur_num, "/", image_num)

          if cur_num >= 2000:
              break


      print("Average BLEU-1 :", np.mean(BLEU[0]/image_num))
      print("Average BLEU-2 :", np.mean(BLEU[1]/image_num))
      print("Average BLEU-3 :", np.mean(BLEU[2]/image_num))
      print("Average BLEU-4 :", np.mean(BLEU[3]/image_num))

      print("Average ROUGE-L-precision :", np.mean(Rouge[0] / image_num))
      print("Average ROUGE-L-recall :", np.mean(Rouge[1] / image_num))
      print("Average ROUGE-L-F1 :", np.mean(Rouge[2] / image_num))


if __name__ == "__main__":
  tf.app.run()
