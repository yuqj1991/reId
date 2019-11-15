import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.python.slim.nets import inception_v3
from tflearn.layers.conv import global_avg_pool

import inception_v3_2
import tensorflow.contrib.slim as slim

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def conv(x_input, shape ,name):
    with tf.variable_scope(name):
        weight = weight_variable(shape)
        bias = weight_variable([shape[-1]])
        result = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_input, weight, strides=[1,1,1,1], padding='SAME'), bias))
        return result


print("\n\n\n")
print("program begin here:")

imgs_ = tf.placeholder(tf.float32, [299,299,3])
imgs = tf.reshape(imgs_, [-1, 299,299,3])
class_num = 1001
is_training_pl = False

g1 = tf.get_default_graph()
with g1.as_default():
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(imgs, num_classes=class_num, is_training=is_training_pl)
    variables = tf.contrib.framework.get_variables_to_restore()
    #print(variables)
    
    unpooling = tf.image.resize_images(end_points['Mixed_7c'], [15, 15])
    print(unpooling.get_shape().as_list())
    
    gavp = global_avg_pool(end_points['Mixed_7c'], name='Global_avg_pooling')
    
    
    
    variables_to_resotre = [v for v in variables if v.name.split('/')[0]=='InceptionV3']
    saver = tf.train.Saver(variables_to_resotre)
    
    

g2 = tf.get_default_graph()
with g2.as_default():
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        end_points_2 = inception_v3_2.inception_v3(imgs, num_classes=class_num, is_training=is_training_pl, scope='InceptionV3_2')
        
        print('Mixed_7c',end_points['Mixed_7c'].get_shape().as_list())
    
    W_conv1 = weight_variable([3,3, 2048, 2048])
    conv1 = tf.nn.atrous_conv2d(end_points_2['Mixed_7c'], W_conv1, rate = 3, padding = 'SAME', name='conv1')
    
    W_conv2 = weight_variable([3,3, 2048, 2048])
    conv2 = tf.nn.atrous_conv2d(conv1, W_conv2, rate = 6, padding = 'SAME', name='conv2')
    
    W_conv3 = weight_variable([3,3, 2048, 2048])
    conv3 = tf.nn.atrous_conv2d(conv2, W_conv3, rate = 9, padding = 'SAME', name='conv3')
    
    W_conv4 = weight_variable([3,3, 2048, 2048])
    conv4 = tf.nn.atrous_conv2d(conv3, W_conv3, rate = 12, padding = 'SAME', name='conv4')
    
    print(conv1.get_shape().as_list())
    print(conv2.get_shape().as_list())
    print(conv3.get_shape().as_list())
    print(conv4.get_shape().as_list())
    
    node = [conv1, conv2, conv3, conv4]
    merge = tf.concat(node, 3)
    merge_conv = tf.nn.softmax(conv(merge, [1,1,4*2048, 5], name='merge_conv'))
    
    
    print(merge_conv.get_shape().as_list())
    size_x, size_y, size_z = unpooling.get_shape().as_list()[1:]
    unpooling_ = tf.reshape(unpooling, [-1, size_x*size_y, size_z])
    size_x, size_y, size_z = merge_conv.get_shape().as_list()[1:]
    merge_conv_ = tf.reshape(merge_conv, [-1, size_x*size_y, size_z])
    
    fin_feature = tf.matmul(merge_conv_, unpooling_, adjoint_a = True)
    print(fin_feature.get_shape().as_list())
        
    #variables = tf.contrib.framework.get_variables_to_restore()
    #print(variables)
    #variables_to_resotre = [v for v in variables if v.name.split('/')[0]=='InceptionV3_2']
    #saver = tf.train.Saver(variables_to_resotre)

'''
with tf.variable_scope('normal_inception'):
    with slim.arg_scope(inception_v3.inception_v3_arg_scope()):
        logits, end_points = inception_v3.inception_v3(imgs, num_classes=class_num, is_training=is_training_pl)
    variables = tf.contrib.framework.get_variables_to_restore()
    print(variables)
    variables_to_resotre = [v for v in variables if v.name.split('/')[0]=='normal_inception' and v.name.split('/')[1]=='InceptionV3']
    saver = tf.train.Saver(variables_to_resotre)

with tf.variable_scope('inception_star'):
    with slim.arg_scope(inception_v3_2.inception_v3_arg_scope()):
        logits2, end_points2 = inception_v3_2.inception_v3(imgs, num_classes=class_num, is_training=is_training_pl, scope = 'InceptionV3_star')
        
'''

#variables = tf.contrib.framework.get_variables_to_restore()
#print(variables)
#variables_to_resotre = [v for v in variables if v.name.split('/')[0]=='normal_inception/InceptionV3']



#saver = tf.train.Saver(variables_to_resotre)

with tf.Session(graph=g2) as sess:
    saver.restore(sess, "./inception_v3.ckpt")
    init = tf.global_variables_initializer()
    sess.run(init)
    img = np.ones([299,299,3])
    #img_ = tf.reshape(img, [1,299,299,3])
    feed_dict = {imgs_:img}
    net = sess.run(end_points, feed_dict=feed_dict)
    
    print(net.keys())
    
#with tf.Session(graph=g1) as sess:
    
    
print("program finish here\n")
    
