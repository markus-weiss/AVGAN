import tensorflow as tf
import numpy as np
import os


init_feat_maps = 32

def conv_relu(in_tensor, layer_name, kernel_size, stride, out_chan, trainable=True):
    """ Conv + Relu layer. """
    with tf.variable_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()
        print('conv_relu, in_size: ', in_size)
        tf.add_to_collection('shapes_for_memory', in_tensor)

        strides = [1, stride, stride, 1]
        kernel_shape = [kernel_size, kernel_size, in_size[3], out_chan]

        # conv
        kernel = tf.get_variable('weights', kernel_shape, tf.float32,
                                 tf.contrib.layers.xavier_initializer_conv2d(), trainable=trainable, collections=['wd', 'variables', 'filters'])
        tmp_result = tf.nn.conv2d(in_tensor, kernel, strides, padding='SAME')

        # bias
        biases = tf.get_variable('biases', [kernel_shape[3]], tf.float32,
                                 tf.constant_initializer(0.0001), trainable=trainable, collections=['wd', 'variables', 'biases'])
        out_tensor = tf.nn.bias_add(tmp_result, biases, name='out')
        #tf.summary.histogram("layer_name/weights", kernel)
        #tf.summary.histogram("layer_name/biases", biases)

        return tf.nn.relu(out_tensor)


def fc(in_tensor, layer_name, out_chan, trainable=True):
    """ FC + Relu layer. """
    with tf.variable_scope(layer_name):
        in_size = in_tensor.get_shape().as_list()
        assert len(in_size) == 2, 'Input to a fully connected layer must be a vector.'
        weights_shape = [in_size[1], out_chan]

        # weight matrix
        weights = tf.get_variable('weights', weights_shape, tf.float32,
                                 tf.contrib.layers.xavier_initializer(), trainable=trainable)
        weights = tf.check_numerics(weights, 'weights: %s' % layer_name)

        # bias
        biases = tf.get_variable('biases', [out_chan], tf.float32,
                                 tf.constant_initializer(0.0001), trainable=trainable)
        biases = tf.check_numerics(biases, 'biases: %s' % layer_name)
        #tf.summary.histogram("layer_name/weights", weights)
        #tf.summary.histogram("layer_name/biases", biases)

        out_tensor = tf.matmul(in_tensor, weights) + biases
        return out_tensor


def build_img_features(input_imgs, img_feat_size, evaluation=False, trainable=True):
    #tf.reset_default_graph()
    """ Creates the VGG 16 network. """
#    s = input_imgs.get_shape().as_list()
#    print('shape input_imgs in build_img_features: ', s)
    batch_size = tf.shape(input_imgs)[0] 
    relu1_1 = conv_relu(input_imgs, "conv1_1", kernel_size=3, stride=1, out_chan=init_feat_maps, trainable=trainable)
    relu1_2 = conv_relu(relu1_1, "conv1_2", kernel_size=3, stride=1, out_chan=init_feat_maps, trainable=trainable)
    pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

    relu2_1 = conv_relu(pool1, "conv2_1", kernel_size=3, stride=1, out_chan=2*init_feat_maps, trainable=trainable)
    relu2_2 = conv_relu(relu2_1, "conv2_2", kernel_size=3, stride=1, out_chan=2*init_feat_maps, trainable=trainable)
    pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    relu3_1 = conv_relu(pool2, "conv3_1", kernel_size=3, stride=1, out_chan=4*init_feat_maps, trainable=trainable)
    relu3_2 = conv_relu(relu3_1, "conv3_2", kernel_size=3, stride=1, out_chan=4*init_feat_maps, trainable=trainable)
    relu3_3 = conv_relu(relu3_2, "conv3_3", kernel_size=3, stride=1, out_chan=4*init_feat_maps, trainable=trainable)
    pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

    relu4_1 = conv_relu(pool3, "conv4_1", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
    relu4_2 = conv_relu(relu4_1, "conv4_2", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
    relu4_3 = conv_relu(relu4_2, "conv4_3", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
    pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    relu5_1 = conv_relu(pool4, "conv5_1", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
    relu5_2 = conv_relu(relu5_1, "conv5_2", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
    relu5_3 = conv_relu(relu5_2, "conv5_3", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
    pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

    #pool5 = tf.reshape(pool5, [s[0], -1])
    size_pool5 = pool5.get_shape().as_list()
    print('pool 5: ', size_pool5)
    print('shape 0: ', batch_size)
    pool5 = tf.reshape(pool5,[ batch_size, size_pool5[1]*size_pool5[2]*size_pool5[3]])
    fc6 = fc(pool5, "fc6", out_chan=img_feat_size, trainable=trainable)
    relu6 = tf.nn.relu(fc6)
    if not evaluation:
        relu6 = tf.nn.dropout(relu6, 0.5)

    fc7 = fc(relu6, "fc7", out_chan=img_feat_size, trainable=trainable)
    relu7 = tf.nn.relu(fc7)
    if not evaluation:
        relu7 = tf.nn.dropout(relu7, 0.5)

    # final classification layer
    fc8 = fc(relu7, "fc8", out_chan=img_feat_size, trainable=trainable)
    print('fc8 ', fc8)
    return fc8, fc7

def build_img_features_stream(stream, input_imgs,trainable=True):
    #tf.reset_default_graph()
    """ Creates the VGG 16 network. """
#    s = input_imgs.get_shape().as_list()
#    print('shape input_imgs in build_img_features: ', s)
    with tf.variable_scope(stream):
        batch_size = tf.shape(input_imgs)[0] 
        relu1_1 = conv_relu(input_imgs, "conv1_1", kernel_size=3, stride=1, out_chan=init_feat_maps, trainable=trainable)
        relu1_2 = conv_relu(relu1_1, "conv1_2", kernel_size=3, stride=1, out_chan=init_feat_maps, trainable=trainable)
        pool1 = tf.nn.max_pool(relu1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    
        relu2_1 = conv_relu(pool1, "conv2_1", kernel_size=3, stride=1, out_chan=2*init_feat_maps, trainable=trainable)
        relu2_2 = conv_relu(relu2_1, "conv2_2", kernel_size=3, stride=1, out_chan=2*init_feat_maps, trainable=trainable)
        pool2 = tf.nn.max_pool(relu2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')
    
        relu3_1 = conv_relu(pool2, "conv3_1", kernel_size=3, stride=1, out_chan=4*init_feat_maps, trainable=trainable)
        relu3_2 = conv_relu(relu3_1, "conv3_2", kernel_size=3, stride=1, out_chan=4*init_feat_maps, trainable=trainable)
        relu3_3 = conv_relu(relu3_2, "conv3_3", kernel_size=3, stride=1, out_chan=4*init_feat_maps, trainable=trainable)
        pool3 = tf.nn.max_pool(relu3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')
    
        relu4_1 = conv_relu(pool3, "conv4_1", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
        relu4_2 = conv_relu(relu4_1, "conv4_2", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
        relu4_3 = conv_relu(relu4_2, "conv4_3", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
        pool4 = tf.nn.max_pool(relu4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')
    
        relu5_1 = conv_relu(pool4, "conv5_1", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
        relu5_2 = conv_relu(relu5_1, "conv5_2", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
        relu5_3 = conv_relu(relu5_2, "conv5_3", kernel_size=3, stride=1, out_chan=8*init_feat_maps, trainable=trainable)
        pool5 = tf.nn.max_pool(relu5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')
    
        #pool5 = tf.reshape(pool5, [s[0], -1])
        size_pool5 = pool5.get_shape().as_list()
        print('pool 5: ', size_pool5)
        print('shape 0: ', batch_size)
        pool5 = tf.reshape(pool5,[ batch_size, size_pool5[1]*size_pool5[2]*size_pool5[3]])       
        return pool5

def average_stream(pool5_img, pool5_flo, img_feat_size, evaluation=False, trainable=True):
    avg_pool5 = (pool5_img+pool5_flo)/2
    fc6 = fc(avg_pool5, "fc6", out_chan=img_feat_size, trainable=trainable)
    relu6 = tf.nn.relu(fc6)
    if not evaluation:
        relu6 = tf.nn.dropout(relu6, 0.5)

    fc7 = fc(relu6, "fc7", out_chan=img_feat_size, trainable=trainable)
    relu7 = tf.nn.relu(fc7)
    if not evaluation:
        relu7 = tf.nn.dropout(relu7, 0.5)

    # final classification layer
    fc8 = fc(relu7, "fc8", out_chan=img_feat_size, trainable=trainable)
    print('fc8 ', fc8)
    return fc8

def build_fc(pool5, img_feat_size, evaluation=False, trainable=True):
    fc6 = fc(pool5, "fc6", out_chan=img_feat_size, trainable=trainable)
    relu6 = tf.nn.relu(fc6)
    if not evaluation:
        relu6 = tf.nn.dropout(relu6, 0.5)

    fc7 = fc(relu6, "fc7", out_chan=img_feat_size, trainable=trainable)
    relu7 = tf.nn.relu(fc7)
    if not evaluation:
        relu7 = tf.nn.dropout(relu7, 0.5)
    return relu7