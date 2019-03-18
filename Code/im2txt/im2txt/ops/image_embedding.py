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

"""Image embedding ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np

from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3_base
from nets.inception_v4 import inception_v4_base

from tensorflow.contrib.layers import batch_norm, flatten
from tensorflow.contrib.framework import arg_scope

slim = tf.contrib.slim


def inception_v3(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV3"):
  """Builds an Inception V3 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v3 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV3", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v3_base(images, scope=scope)
        with tf.variable_scope("logits"):
            # 32*8*8*2048
            shape = net.get_shape()
            # 32*1*1*2048
            net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
            net = slim.dropout(
                  net,
                  keep_prob=dropout_keep_prob,
                  is_training=is_inception_model_training,
                  scope="dropout")
            net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net

def inception_v4(images,
                 trainable=True,
                 is_training=True,
                 weight_decay=0.00004,
                 stddev=0.1,
                 dropout_keep_prob=0.8,
                 use_batch_norm=True,
                 batch_norm_params=None,
                 add_summaries=True,
                 scope="InceptionV4"):
  """Builds an Inception V4 subgraph for image embeddings.

  Args:
    images: A float32 Tensor of shape [batch, height, width, channels].
    trainable: Whether the inception submodel should be trainable or not.
    is_training: Boolean indicating training mode or not.
    weight_decay: Coefficient for weight regularization.
    stddev: The standard deviation of the trunctated normal weight initializer.
    dropout_keep_prob: Dropout keep probability.
    use_batch_norm: Whether to use batch normalization.
    batch_norm_params: Parameters for batch normalization. See
      tf.contrib.layers.batch_norm for details.
    add_summaries: Whether to add activation summaries.
    scope: Optional Variable scope.

  Returns:
    end_points: A dictionary of activations from inception_v4 layers.
  """
  # Only consider the inception model to be in training mode if it's trainable.
  is_inception_model_training = trainable and is_training

  if use_batch_norm:
    # Default parameters for batch normalization.
    if not batch_norm_params:
      batch_norm_params = {
          "is_training": is_inception_model_training,
          "trainable": trainable,
          # Decay for the moving averages.
          "decay": 0.9997,
          # Epsilon to prevent 0s in variance.
          "epsilon": 0.001,
          # Collection containing the moving mean and moving variance.
          "variables_collections": {
              "beta": None,
              "gamma": None,
              "moving_mean": ["moving_vars"],
              "moving_variance": ["moving_vars"],
          }
      }
  else:
    batch_norm_params = None

  if trainable:
    weights_regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
  else:
    weights_regularizer = None

  with tf.variable_scope(scope, "InceptionV4", [images]) as scope:
    with slim.arg_scope(
        [slim.conv2d, slim.fully_connected],
        weights_regularizer=weights_regularizer,
        trainable=trainable):
      with slim.arg_scope(
          [slim.conv2d],
          weights_initializer=tf.truncated_normal_initializer(stddev=stddev),
          activation_fn=tf.nn.relu,
          normalizer_fn=slim.batch_norm,
          normalizer_params=batch_norm_params):
        net, end_points = inception_v4_base(images, scope=scope)
        with tf.variable_scope("logits"):
            # 32*8*8*1536
            shape = net.get_shape()
            # 32*1*1*1536
            net = slim.avg_pool2d(net, shape[1:3], padding="VALID", scope="pool")
            net = slim.dropout(
                  net,
                  keep_prob=dropout_keep_prob,
                  is_training=is_inception_model_training,
                  scope="dropout")
            net = slim.flatten(net, scope="flatten")

  # Add summaries.
  if add_summaries:
    for v in end_points.values():
      tf.contrib.layers.summaries.summarize_activation(v)

  return net


# dense-121
class DenseNet():
    def __init__(self, x, nb_blocks, filters, training, dropout_rate=0.2, scope='Densenet'):
        self.nb_blocks = nb_blocks
        self.filters = filters
        self.training = training  # tf.placeholder(tf.bool, name='dn_istraining')
        self.dropout_rate = dropout_rate
        with tf.variable_scope(scope):
            self.model = self.Dense_net(x)

    #  growth_k = 24
    #  nb_block = 2
    dropout_rate = 0.2

    def conv_layer(self, input, filter, kernel, stride=1, layer_name="conv"):
        with tf.name_scope(layer_name):
            network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride,
                                       padding='SAME')
            return network

    def Global_Average_Pooling(self, x, stride=1):
        width = np.shape(x)[1]
        height = np.shape(x)[2]
        pool_size = [width, height]
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)

    def Batch_Normalization(self, x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True):
            return batch_norm(inputs=x, is_training=training, reuse=None)

    def Drop_out(self, x, rate, training):
        return tf.layers.dropout(inputs=x, rate=rate, training=training)

    def Relu(self, x):
        return tf.nn.relu(x)

    def Average_pooling(self, x, pool_size=[2, 2], stride=2, padding='VALID'):
        return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Max_Pooling(self, x, pool_size=[3, 3], stride=2, padding='VALID'):
        return tf.layers.max_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

    def Concatenation(self, layers):
        return tf.concat(layers, axis=3)

    def bottleneck_layer(self, x, scope):
        # print(x)
        with tf.name_scope(scope):
            x = self.Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = self.Relu(x)
            x = self.conv_layer(x, filter=4 * self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = self.Drop_out(x, rate=self.dropout_rate, training=self.training)

            x = self.Batch_Normalization(x, training=self.training, scope=scope + '_batch2')
            x = self.Relu(x)
            x = self.conv_layer(x, filter=self.filters, kernel=[3, 3], layer_name=scope + '_conv2')
            x = self.Drop_out(x, rate=self.dropout_rate, training=self.training)

            # print(x)

            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            x = self.Batch_Normalization(x, training=self.training, scope=scope + '_batch1')
            x = self.Relu(x)
            x = self.conv_layer(x, filter=self.filters, kernel=[1, 1], layer_name=scope + '_conv1')
            x = self.Drop_out(x, rate=self.dropout_rate, training=self.training)
            x = self.Average_pooling(x, pool_size=[2, 2], stride=2)

            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = self.Concatenation(layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = self.Concatenation(layers_concat)

            return x

    def Dense_net(self, input_x):
        # input_x 299*299*3
        x = self.conv_layer(input_x, filter= 2 * self.filters, kernel=[7, 7], stride=2, layer_name='conv0')
        # 150*150*48
        # x = Max_Pooling(x, pool_size=[3,3], stride=2)

        x = self.dense_block(input_x=x, nb_layers=6, layer_name='dense_1')  # 192
        x = self.transition_layer(x, scope='trans_1')
        # 75*75*24
        x = self.dense_block(input_x=x, nb_layers=12, layer_name='dense_2')  # 75*75*312
        x = self.transition_layer(x, scope='trans_2')
        # 37*37*24
        x = self.dense_block(input_x=x, nb_layers=48, layer_name='dense_3')  # 1176
        x = self.transition_layer(x, scope='trans_3')
        # 18*18*24
        x = self.dense_block(input_x=x, nb_layers=32, layer_name='dense_4')
        # x = self.transition_layer(x, scope='trans_4')
        # 18*18*792
        with tf.name_scope('final_layer'):
            x = self.Batch_Normalization(x, training=self.training, scope='final_layer' + '_batch1')
            x = self.Relu(x)
            x = self.conv_layer(x, filter=512, kernel=[1, 1], layer_name='final_layer' + '_conv1')
            x = self.Drop_out(x, rate=self.dropout_rate, training=self.training)

        # 100 Layer
        x = self.Batch_Normalization(x, training=self.training, scope='linear_batch')
        x = self.Relu(x)
        x = self.Global_Average_Pooling(x)
        x = flatten(x)
        return x
