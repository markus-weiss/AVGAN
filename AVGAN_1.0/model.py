# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 20:19:06 2018

@author: LaRu
"""
import tensorflow as tf


def build_model(input_img, label, train_mode, drop_out, lr, batch_size):

    in_size = input_img.get_shape().as_list()
    print('input shape: ', in_size)
    with tf.variable_scope('conv_relu1', reuse=tf.AUTO_REUSE):
        kernel1 = tf.get_variable('weights', [3, 3, in_size[3], 32], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        conv1 = tf.nn.conv2d(input = input_img, filter=kernel1, strides = [1,1,1,1],padding = 'SAME')
        biases = tf.get_variable('biases', [32], tf.float32, tf.constant_initializer(0.0001))
        conv1_bias = tf.nn.bias_add(conv1, biases, name='out')
        relu1 = tf.nn.relu(conv1_bias)
    #repeat the same operations again, initialize new training weights with same shape
    with tf.variable_scope('conv_relu2', reuse=tf.AUTO_REUSE):
        kernel2 = tf.get_variable('weights', [3, 3, 32, 32], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        conv2 = tf.nn.conv2d(input = relu1, filter=kernel2, strides = [1,1,1,1],padding = 'SAME')
        biases2 = tf.get_variable('biases', [32], tf.float32, tf.constant_initializer(0.0001))
        conv2_bias = tf.nn.bias_add(conv2, biases2, name='out')
        relu2 = tf.nn.relu(conv2_bias)
    #subsample by maxpool
    with tf.variable_scope('pool2', reuse=tf.AUTO_REUSE) :
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')# ==>16x16x32
    #conv, bias, relu
    with tf.variable_scope('conv_relu3', reuse=tf.AUTO_REUSE):
        kernel3 = tf.get_variable('weights', [3, 3, 32, 64], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        conv3 = tf.nn.conv2d(input = pool2, filter=kernel3, strides = [1,1,1,1],padding = 'SAME')
        biases3 = tf.get_variable('biases', [64], tf.float32, tf.constant_initializer(0.0001))
        conv3_bias = tf.nn.bias_add(conv3, biases3, name='out')
        relu3 = tf.nn.relu(conv3_bias)# 16x16x64
    #subsample by maxpool
    with tf.variable_scope('pool3', reuse=tf.AUTO_REUSE):
        pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')#8x8x64
    
        pool3_reshaped = tf.reshape(pool3,[batch_size, 8*8*64])
    with tf.variable_scope('fc_relu', reuse=tf.AUTO_REUSE):
        w_fc = tf.get_variable('weights', [8*8*64, 512], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        biases_fc = tf.get_variable('biases', [512], tf.float32, tf.constant_initializer(0.0001))
        fc = tf.matmul(pool3_reshaped, w_fc)+biases_fc
        relu_fc = tf.nn.relu(fc)
    if train_mode:
        relu_fc = tf.nn.dropout(relu_fc, drop_out)
    
    with tf.variable_scope('fc_relu5', reuse=tf.AUTO_REUSE):
        w_fc2 = tf.get_variable('weights', [512,10], tf.float32, tf.contrib.layers.xavier_initializer_conv2d())
        biases_fc2 = tf.get_variable('biases', [10], tf.float32, tf.constant_initializer(0.0001))
        logits = tf.matmul(relu_fc, w_fc2)+biases_fc2
    loss = tf.losses.sparse_softmax_cross_entropy(label, logits)
    #axis = 1 for argmax
    argmax = tf.argmax(tf.nn.softmax(logits),1)
    equal_no = tf.cast(tf.equal(tf.cast(label, tf.int64), argmax), tf.float32)
    #accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.cast(label, tf.int64), argmax), tf.float32))
    acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(label, tf.int64), argmax), tf.float32))
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    if train_mode:
        # Optimization
        optimizer = tf.train.AdamOptimizer(lr).minimize(loss, global_step)    
        #optimizer = tf.train.RMSPropOptimizer(lr,decay=1e-6).minimize(loss, global_step)
        tf.summary.scalar('loss', loss)
        tf.summary.histogram('histogram loss', loss)

    tf.summary.scalar('accuracy', acc)
    summary_op = tf.summary.merge_all()
    if train_mode:
        return optimizer, global_step, loss, acc, summary_op, equal_no
    else:
        return global_step,  acc,summary_op