# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains the definition for inception v3 classification network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import inception_utils

slim = tf.contrib.slim
trunc_normal = lambda stddev: tf.truncated_normal_initializer(0.0, stddev)


# 定义了 inception 网络从输入到输出前面一层的网络结构，
# 在 inception_v3 的主体的开头部分调用
def inception_v3_base(inputs,
                      final_endpoint='Mixed_7c',
                      min_depth=16,
                      depth_multiplier=1.0,
                      scope=None):
  """Inception model from http://arxiv.org/abs/1512.00567.
  Constructs an Inception v3 network from inputs to the given final endpoint.
  This method can construct the network up to the final inception block
  Mixed_7c.
  Note that the names of the layers in the paper do not correspond to the names
  of the endpoints registered by this function although they build the same
  network.
  Here is a mapping from the old_names to the new names:
  Old name          | New name          网络层的名称映射
  =======================================

  从这里可以很清晰的看到网络的结构，前面的7个普通网络层，其中包括5个卷积层和2个池化层，
  后接三个 mixed 模块，最后是 inception_v3 主体中的输出层

  conv0             | Conv2d_1a_3x3
  conv1             | Conv2d_2a_3x3
  conv2             | Conv2d_2b_3x3
  pool1             | MaxPool_3a_3x3
  conv3             | Conv2d_3b_1x1
  conv4             | Conv2d_4a_3x3
  pool2             | MaxPool_5a_3x3

  mixed_35x35x256a  | Mixed_5b
  mixed_35x35x288a  | Mixed_5c
  mixed_35x35x288b  | Mixed_5d

  mixed_17x17x768a  | Mixed_6a
  mixed_17x17x768b  | Mixed_6b
  mixed_17x17x768c  | Mixed_6c
  mixed_17x17x768d  | Mixed_6d
  mixed_17x17x768e  | Mixed_6e

  mixed_8x8x1280a   | Mixed_7a
  mixed_8x8x2048a   | Mixed_7b
  mixed_8x8x2048b   | Mixed_7c

  Args:
    输入的 tensor, 形状为 [batch_size, height, width, channels] 
    inputs: a tensor of size [batch_size, height, width, channels].

    这个 base 模块的终点节点名称，可以是该网络的任意一层
    final_endpoint: specifies the endpoint to construct the network up to. It
      can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
      'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
      'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
      'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].

    卷积层的最小通道数， 这里设置为16
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.

    通道数的倍数，设置范围为0-1，可以设置小点，减少计算量，默认为1
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.
    
    模型的变量空间
    scope: Optional variable_scope.

  Returns:
    网络输出的张量
    tensor_out: output tensor corresponding to the final_endpoint.
    辅助输出
    end_points: a set of activations for external use, for example summaries or
                losses.
  Raises:
    ValueError: if final_endpoint is not set to one of the predefined values,
                or depth_multiplier <= 0
  """
  # end_points will collect relevant activations for external use, for example
  # summaries or losses.
  end_points = {}

  # 通道数倍数不能小于0，否则报错
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')

  # 每个卷积层的通道数计算公式， 比如，倍数设置为2，输入32，则输出通道数为64，如果小于16，则输出16  
  depth = lambda d: max(int(d * depth_multiplier), min_depth)

  with tf.variable_scope(scope, 'InceptionV3', [inputs]):
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='VALID'):          # padding='VALID'！！！
      # 299 x 299 x 3   输入图片的尺寸，因为stride为2，所以输出尺寸减半
      end_point = 'Conv2d_1a_3x3'
      net = slim.conv2d(inputs, depth(32), [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net  # 增加辅助节点信息
      # 如果网络终点层在这里，就在这里停止，并且返回，后续都是如此
      if end_point == final_endpoint: return net, end_points
      # 149 x 149 x 32   默认的padding方式为 valid，不填充，导致尺寸减少了2
      end_point = 'Conv2d_2a_3x3'
      net = slim.conv2d(net, depth(32), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 147 x 147 x 32    padding='SAME'   尺寸不变
      end_point = 'Conv2d_2b_3x3'
      net = slim.conv2d(net, depth(64), [3, 3], padding='SAME', scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 147 x 147 x 64     池化层， stride=2 ，尺寸减半
      end_point = 'MaxPool_3a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 73 x 73 x 64       卷积核为[1, 1]，尺寸不变 
      end_point = 'Conv2d_3b_1x1'
      net = slim.conv2d(net, depth(80), [1, 1], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 73 x 73 x 80.      默认的padding方式为 valid，不填充，导致尺寸减少了2
      end_point = 'Conv2d_4a_3x3'
      net = slim.conv2d(net, depth(192), [3, 3], scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 71 x 71 x 192.      池化层， stride=2 ，尺寸减半
      end_point = 'MaxPool_5a_3x3'
      net = slim.max_pool2d(net, [3, 3], stride=2, scope=end_point)
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      # 35 x 35 x 192.    前面7层最后的输出尺寸

    # Inception blocks   inception 的并联模块
    with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                        stride=1, padding='SAME'):   
                        # 注意这里的参数都是 stride=1, padding='SAME'  padding改变了
      # mixed: 35 x 35 x 256.
      end_point = 'Mixed_5b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          # 这里得到35 x 35 x 64
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          # 这里得到 35 x 35 x 64
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          # 这里得到 35 x 35 x 96
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          # 这里得到 35 x 35 x 32
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(32), [1, 1],
                                 scope='Conv2d_0b_1x1')
          # 最后第三维度相加，64+64+96+32=256， 最后输出的尺寸为 35 x 35 x 256
          # 这里非常明显的体现了 inception 的并联思路， 1*1，1*1和5*5， 1*1和3*3和3*3，平均池化和1*1，
          # 这里四个支路的并联，最后拼接在一起
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_1: 35 x 35 x 288.    这里和上面的模块就是一个思路了，简单的做一个注释
      end_point = 'Mixed_5c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          # 这里得到 35 x 35 x 64
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          # 这里得到 35 x 35 x 64
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0b_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
          # 这里得到 35 x 35 x 96
          branch_2 = slim.conv2d(net, depth(64), [1, 1],
                                 scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          # 这里得到 35 x 35 x 64   比之前的32多了32
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
          # 最后第三维度相加，64+64+96+64=288， 最后输出的尺寸为 35 x 35 x 288
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_2: 35 x 35 x 288.   这个模块和上面的模块一模一样
      end_point = 'Mixed_5d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(48), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(64), [5, 5],
                                 scope='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_2 = slim.conv2d(branch_2, depth(96), [3, 3],
                                 scope='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(64), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_3: 17 x 17 x 768.     新的并联结构模块，上面的最后输出的尺寸为 35 x 35 x 288
      end_point = 'Mixed_6a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          # 这里得到 17 x 17 x 384  因为 stride=2
          branch_0 = slim.conv2d(net, depth(384), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
          # 这里得到 17 x 17 x 96  因为 stride=2
          branch_1 = slim.conv2d(net, depth(64), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3],
                                 scope='Conv2d_0b_3x3')
          branch_1 = slim.conv2d(branch_1, depth(96), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
          # 这里得到 17 x 17 x 288  因为 stride=2
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
          # 384+96+288=768
          # 之前的并联思路， 1*1，1*1和5*5， 1*1和3*3和3*3，平均池化和1*1，   35 x 35 x 288
          # 这里的并联思路， 3*3，          1*1和3*3和3*3，最大池化，stride=2，17 x 17 x 768， 通道数增加到768 
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed4: 17 x 17 x 768.
      # 之前的并联思路， 1*1，1*1和5*5， 1*1和3*3和3*3，平均池化和1*1，   35 x 35 x 288
      # 之前的并联思路， 3*3，          1*1和3*3和3*3，最大池化，stride=2，17 x 17 x 768， 通道数增加到768 
      # 这里的并联思路， 1*1，1*1和1*7和7*1， 1*1和1*7和7*1和1*7和7*1，平均池化和1*1， 17 x 17 x 768， 通道数增加到768       
      end_point = 'Mixed_6b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(128), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(128), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(128), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(128), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_5: 17 x 17 x 768.    这个和上面的模块一模一样
      end_point = 'Mixed_6c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      
      # mixed_6: 17 x 17 x 768.      这个和上面的模块一模一样
      end_point = 'Mixed_6d'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(160), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(160), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(160), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(160), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_7: 17 x 17 x 768.     这个和上面的模块一模一样
      end_point = 'Mixed_6e'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0b_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0c_1x7')
          branch_2 = slim.conv2d(branch_2, depth(192), [7, 1],
                                 scope='Conv2d_0d_7x1')
          branch_2 = slim.conv2d(branch_2, depth(192), [1, 7],
                                 scope='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(branch_3, depth(192), [1, 1],
                                 scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_8: 8 x 8 x 1280.  
      # 之前的并联思路， 1*1，1*1和5*5， 1*1和3*3和3*3，平均池化和1*1，   35 x 35 x 288
      # 之前的并联思路， 3*3，          1*1和3*3和3*3，最大池化，stride=2，17 x 17 x 768， 通道数增加到768 
      # 之前的并联思路， 1*1，1*1和1*7和7*1， 1*1和1*7和7*1和1*7和7*1，平均池化和1*1， 17 x 17 x 768， 通道数增加到768      end_point = 'Mixed_7a'
      # 这里的并联思路， 1*1和3*3，1*1和1*7和7*1和3*3， 最大池化，8 x 8 x 1280 通道数增加到 1280      
      end_point = 'Mixed_7a'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_0 = slim.conv2d(branch_0, depth(320), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(192), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [1, 7],
                                 scope='Conv2d_0b_1x7')
          branch_1 = slim.conv2d(branch_1, depth(192), [7, 1],
                                 scope='Conv2d_0c_7x1')
          branch_1 = slim.conv2d(branch_1, depth(192), [3, 3], stride=2,
                                 padding='VALID', scope='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.max_pool2d(net, [3, 3], stride=2, padding='VALID',
                                     scope='MaxPool_1a_3x3')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
      
      # mixed_9: 8 x 8 x 2048.
      # 之前的并联思路， 1*1，1*1和5*5， 1*1和3*3和3*3，平均池化和1*1，   35 x 35 x 288
      # 之前的并联思路， 3*3，          1*1和3*3和3*3，最大池化，stride=2，17 x 17 x 768， 通道数增加到768 
      # 之前的并联思路， 1*1，1*1和1*7和7*1， 1*1和1*7和7*1和1*7和7*1，平均池化和1*1， 17 x 17 x 768， 通道数增加到768      end_point = 'Mixed_7a'
      # 之前的并联思路， 1*1和3*3，1*1和1*7和7*1和3*3， 最大池化，8 x 8 x 1280 通道数增加到 1280        
      # 这里的并联思路， 1*1，1*1和1*3和3*1，这里有个 net in net 的又一次小的并联结构，平均池化和1*1， 
      # 最后得到 8 x 8 x 2048， 通道数增加到 2048      
      end_point = 'Mixed_7b'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0b_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points

      # mixed_10: 8 x 8 x 2048.    和上面一样
      end_point = 'Mixed_7c'
      with tf.variable_scope(end_point):
        with tf.variable_scope('Branch_0'):
          branch_0 = slim.conv2d(net, depth(320), [1, 1], scope='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
          branch_1 = slim.conv2d(net, depth(384), [1, 1], scope='Conv2d_0a_1x1')
          branch_1 = tf.concat(axis=3, values=[
              slim.conv2d(branch_1, depth(384), [1, 3], scope='Conv2d_0b_1x3'),
              slim.conv2d(branch_1, depth(384), [3, 1], scope='Conv2d_0c_3x1')])
        with tf.variable_scope('Branch_2'):
          branch_2 = slim.conv2d(net, depth(448), [1, 1], scope='Conv2d_0a_1x1')
          branch_2 = slim.conv2d(
              branch_2, depth(384), [3, 3], scope='Conv2d_0b_3x3')
          branch_2 = tf.concat(axis=3, values=[
              slim.conv2d(branch_2, depth(384), [1, 3], scope='Conv2d_0c_1x3'),
              slim.conv2d(branch_2, depth(384), [3, 1], scope='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
          branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
          branch_3 = slim.conv2d(
              branch_3, depth(192), [1, 1], scope='Conv2d_0b_1x1')
        net = tf.concat(axis=3, values=[branch_0, branch_1, branch_2, branch_3])
      end_points[end_point] = net
      if end_point == final_endpoint: return net, end_points
    raise ValueError('Unknown final endpoint %s' % final_endpoint) 
    # base 模块最后输出 8 x 8 x 2048


# inception_v3 结构的主体部分
def inception_v3(inputs,
                 num_classes=1000,
                 is_training=True,
                 dropout_keep_prob=0.8,
                 min_depth=16,
                 depth_multiplier=1.0,
                 prediction_fn=slim.softmax,
                 spatial_squeeze=True,
                 reuse=None,
                 create_aux_logits=True,
                 scope='InceptionV3',
                 global_pool=False):
  """Inception model from http://arxiv.org/abs/1512.00567.
  "Rethinking the Inception Architecture for Computer Vision"
  Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
  Zbigniew Wojna.
  With the default arguments this method constructs the exact model defined in
  the paper. However, one can experiment with variations of the inception_v3
  network by changing arguments dropout_keep_prob, min_depth and
  depth_multiplier.
  The default image size used to train this network is 299x299.

  这段话意思是使用默认的参数可以得到和论文中提到的inception_v3一样的结构，但是你也可以通过调整
  dropout_keep_prob, min_depth ，depth_multiplier 参数来得到不同的结构变体；
  另外，默认的输入图片尺寸为 299x299

  Args:
    输入尺寸
    inputs: a tensor of size [batch_size, height, width, channels].
    最终输出的分类数量
    num_classes: number of predicted classes. If 0 or None, the logits layer
      is omitted and the input features to the logits layer (before dropout)
      are returned instead.

    训练模式还是测试模式的开关
    is_training: whether is training or not.

    dropout 比例
    dropout_keep_prob: the percentage of activation values that are retained.

    最小通道数
    min_depth: Minimum depth value (number of channels) for all convolution ops.
      Enforced when depth_multiplier < 1, and not an active constraint when
      depth_multiplier >= 1.

    通道数倍数
    depth_multiplier: Float multiplier for the depth (number of channels)
      for all convolution ops. The value must be greater than zero. Typical
      usage will be to set this value in (0, 1) to reduce the number of
      parameters or computation cost of the model.

    logits 之后的输出经过的函数，默认为 softmax
    prediction_fn: a function to get predictions out of logits.

    是否压缩去掉数目为1的维度
    spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
        shape [B, 1, 1, C], where B is batch_size and C is number of classes.
    
    变量是否重复使用
    reuse: whether or not the network and its variables should be reused. To be
      able to reuse 'scope' must be given.

    是否创建辅助logits 节点
    create_aux_logits: Whether to create the auxiliary logits.
    scope: Optional variable_scope.

    是否使用全局池化，默认不使用，使用的情况下，任意大小的输入都会被平均，导致输出为1*1
    global_pool: Optional boolean flag to control the avgpooling before the
      logits layer. If false or unset, pooling is done with a fixed window
      that reduces default-sized inputs to 1x1, while larger inputs lead to
      larger outputs. If true, any input size is pooled down to 1x1.
  Returns:
    输出，logits
    net: a Tensor with the logits (pre-softmax activations) if num_classes
      is a non-zero integer, or the non-dropped-out input to the logits layer
      if num_classes is 0 or None.
    网络各个节点的数据
    end_points: a dictionary from components of the network to the corresponding
      activation.
  Raises:
    ValueError: if 'depth_multiplier' is less than or equal to zero.
  """
  if depth_multiplier <= 0:
    raise ValueError('depth_multiplier is not greater than zero.')
  depth = lambda d: max(int(d * depth_multiplier), min_depth)
  
  # 这里开头就使用 base 模块构建了 inception 网络的前面部分结构，在辅助节点之后又构建了网络的尾部
  # 构建了网络的 arg_scope， 一个问题是这里都没有用到 get_variable， 那这个 reuse 参数的影响又体现在哪里呢？
  with tf.variable_scope(scope, 'InceptionV3', [inputs], reuse=reuse) as scope:
    # 设置 slim.batch_norm, slim.dropout 的参数值， dropout 倒是只在最后的输出前面一层用到了一次，
    # 而 batch_norm 则是在每个卷积层都默认使用了，因为事先指定了slim.conv2d 的 normalizer_fn 参数
    with slim.arg_scope([slim.batch_norm, slim.dropout],
                        is_training=is_training):
      net, end_points = inception_v3_base(
          inputs, scope=scope, min_depth=min_depth,
          depth_multiplier=depth_multiplier)

      # Auxiliary Head logits   辅助节点信息，是从 Mixed_6e 拉出来的节点
      if create_aux_logits and num_classes:
        with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d],
                            stride=1, padding='SAME'):
          aux_logits = end_points['Mixed_6e']
          with tf.variable_scope('AuxLogits'):
            aux_logits = slim.avg_pool2d(
                aux_logits, [5, 5], stride=3, padding='VALID',
                scope='AvgPool_1a_5x5')
            aux_logits = slim.conv2d(aux_logits, depth(128), [1, 1],
                                     scope='Conv2d_1b_1x1')

            # Shape of feature map before the final layer.
            kernel_size = _reduced_kernel_size_for_small_input(
                aux_logits, [5, 5])
            aux_logits = slim.conv2d(
                aux_logits, depth(768), kernel_size,
                weights_initializer=trunc_normal(0.01),
                padding='VALID', scope='Conv2d_2a_{}x{}'.format(*kernel_size))
            aux_logits = slim.conv2d(
                aux_logits, num_classes, [1, 1], activation_fn=None,
                normalizer_fn=None, weights_initializer=trunc_normal(0.001),
                scope='Conv2d_2b_1x1')
            if spatial_squeeze:
              aux_logits = tf.squeeze(aux_logits, [1, 2], name='SpatialSqueeze')
            end_points['AuxLogits'] = aux_logits

      # Final pooling and prediction   网络的尾部，base 输出为 8 x 8 x 2048，经过全局平均池化后为 1 x 1 x 2048
      with tf.variable_scope('Logits'):
        if global_pool:
          # Global average pooling.
          net = tf.reduce_mean(net, [1, 2], keep_dims=True, name='GlobalPool')
          end_points['global_pool'] = net
        else:
          # Pooling with a fixed kernel size.   
          # 这里 global_pool 为 T 和 F 效果都是一样的，因为输入 是8 x 8 x 2048， 而这里的平均池化的核大小也是 8*8 
          kernel_size = _reduced_kernel_size_for_small_input(net, [8, 8])
          net = slim.avg_pool2d(net, kernel_size, padding='VALID',
                                scope='AvgPool_1a_{}x{}'.format(*kernel_size))
          end_points['AvgPool_1a'] = net
        if not num_classes:
          return net, end_points
        # 1 x 1 x 2048
        net = slim.dropout(net, keep_prob=dropout_keep_prob, scope='Dropout_1b')
        end_points['PreLogits'] = net
        # 2048      
        # 这里得到分类结果，由1 x 1 x 2048变为 【batch，1，1，num_classes】
        logits = slim.conv2d(net, num_classes, [1, 1], activation_fn=None,
                             normalizer_fn=None, scope='Conv2d_1c_1x1')
        if spatial_squeeze:   # 压缩去掉 1和2 维度
          logits = tf.squeeze(logits, [1, 2], name='SpatialSqueeze')
        # 1000
      end_points['Logits'] = logits
      end_points['Predictions'] = prediction_fn(logits, scope='Predictions')
  return logits, end_points

inception_v3.default_image_size = 299


def _reduced_kernel_size_for_small_input(input_tensor, kernel_size):
  """
  定义网络的卷积核的大小，使它不超过输入的 tensor 的大小，
  在输入的 tensor 尺寸较小的时候发挥作用，

  Define kernel size which is automatically reduced for small input.
  If the shape of the input images is unknown at graph construction time this
  function assumes that the input images are is large enough.
  Args:
    input_tensor: input tensor of size [batch_size, height, width, channels].
    kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]
  Returns:
    a tensor with the kernel size.
  TODO(jrru): Make this function work with unknown shapes. Theoretically, this
  can be done with the code below. Problems are two-fold: (1) If the shape was
  known, it will be lost. (2) inception.slim.ops._two_element_tuple cannot
  handle tensors that define the kernel size.
      shape = tf.shape(input_tensor)
      return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                         tf.minimum(shape[2], kernel_size[1])])
  """
  shape = input_tensor.get_shape().as_list()
  if shape[1] is None or shape[2] is None:
    kernel_size_out = kernel_size
  else:
    kernel_size_out = [min(shape[1], kernel_size[0]),
                       min(shape[2], kernel_size[1])]
  return kernel_size_out


# 这里倒是有点疑问， 在 inception_v3 的主体部分已经设置了 arg_scope ，为什么这里又要再设置一次
inception_v3_arg_scope = inception_utils.inception_arg_scope




# 上面用到的 inception_arg_scope 是一个如下的函数，它返回整个 inception 网络的参数作用域
def inception_arg_scope(weight_decay=0.00004,
                        use_batch_norm=True,
                        batch_norm_decay=0.9997,
                        batch_norm_epsilon=0.001,
                        activation_fn=tf.nn.relu):
  """Defines the default arg scope for inception models.
  Args:
    weight_decay: The weight decay to use for regularizing the model.
    use_batch_norm: "If `True`, batch_norm is applied after each convolution.
    batch_norm_decay: Decay for batch norm moving average.
    batch_norm_epsilon: Small float added to variance to avoid dividing by zero
      in batch norm.
    activation_fn: Activation function for conv2d.
  Returns:
    An `arg_scope` to use for the inception models.
  """
  batch_norm_params = {
      # Decay for the moving averages.
      'decay': batch_norm_decay,
      # epsilon to prevent 0s in variance.
      'epsilon': batch_norm_epsilon,
      # collection containing update_ops.
      'updates_collections': tf.GraphKeys.UPDATE_OPS,
      # use fused batch norm if possible.
      'fused': None,
  }
  # 是否使用BN批标准化操作
  if use_batch_norm:
    normalizer_fn = slim.batch_norm
    normalizer_params = batch_norm_params
  else:
    normalizer_fn = None
    normalizer_params = {}
  # Set weight_decay for weights in Conv and FC layers.
  # 嵌套的给卷积层和全连接层设置参数正则化； 
  # 嵌套的给卷积层设置参数初始化、激活函数、BN批标准化
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      weights_regularizer=slim.l2_regularizer(weight_decay)):
    with slim.arg_scope(
        [slim.conv2d],
        weights_initializer=slim.variance_scaling_initializer(),
        activation_fn=activation_fn,
        normalizer_fn=normalizer_fn,
        normalizer_params=normalizer_params) as sc:
      return sc
