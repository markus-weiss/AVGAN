# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 11:00:33 2018

@author: LaRu
"""
import tensorflow as tf

from tensorflow.python.keras import datasets
import numpy as np
from model import *


#discomment after restart
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

print(x_train.shape)#(50000, 32, 32, 3)
print(y_train.shape, y_train.dtype)
x_train = x_train.astype(np.float32)
y_train = y_train.astype(np.int32)
#print('type: ', x_train.dtype)
x_test = x_test.astype(np.float32)
y_test = y_test.astype(np.int32)
mode =0 # 1=eval, 0 = train,
epochs = 2#accuracy for 25 0.78, test acc 0.59
checkpoints_path = './checkpoints'
#imgs_placeholder = tf.placeholder(x_train.dtype, x_train.shape)
#labels_placeholder = tf.placeholder(y_train.dtype, y_train.shape)
batch_size = 20
lr = 0.0001
keep_prob = .5 #keep
num_iter = np.int(x_train.shape[0]/batch_size)*epochs
print('iter: ', num_iter)
#dataset = tf.data.Dataset.from_tensor_slices((imgs_placeholder, labels_placeholder))
tf.reset_default_graph()
if (mode == 0):
    y_train = np.reshape(y_train,[50000])
    dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))
    # [Other transformations on `dataset`...]
    batched_dataset = dataset.batch(batch_size)
    #
    iterator = batched_dataset.make_initializable_iterator()
    next_element, next_label = iterator.get_next()
    print('img_tensor: ', next_element)
    print('label: ', next_label)
     # conv, bias, relu
    #tf.reset_default_graph()
    optimizer, global_step, loss, accuracy, summary_op, _ = build_model(next_element, next_label, True, keep_prob, lr, batch_size)
    saver = tf.train.Saver()

    train_sess = tf.Session()
    writer = tf.summary.FileWriter('./graphs/convnet', tf.get_default_graph())#train_sess.graph)
    
   
    train_sess.run(tf.global_variables_initializer())

    path = tf.train.latest_checkpoint(checkpoints_path)
        
    if path:
        print('restore path: ', path)
        saver.restore(train_sess,path)
        print('global_step', global_step.eval(train_sess))
    else:
        print('could not restore')
    total_acc = 0
    for i in range(epochs):
        train_sess.run(iterator.initializer)
        j = 0
        acc_per_epoch = 0
        try:
            while True:
                loss_, acc_m, summary_, _, global_step_ = train_sess.run([loss,  accuracy, summary_op, optimizer, global_step])#,feed_dict={imgs_placeholder: x_train, labels_placeholder: y_train}) 
                if (global_step_ % 200 == 0):
                    print ("Loss at global step", global_step_, ":", loss_, acc_m) 
                writer.add_summary(summary_)
                acc_per_epoch += acc_m
                j += 1
        except tf.errors.OutOfRangeError:
            print('Epoch finished')
            acc_per_epoch = acc_per_epoch/j
            total_acc += acc_per_epoch
    print('total_acc_train: ', total_acc/epochs)
    saver.save(train_sess, checkpoints_path+'/ckpts', global_step = global_step)    
    writer.close()
    train_sess.close()
if (mode == 1):
    y_test = np.reshape(y_test, [10000])
    dataset = tf.data.Dataset.from_tensor_slices((x_test,y_test))
    print('test shape: ', x_test.shape, y_test)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    next_element, next_label = iterator.get_next()
     # conv, bias, relu
    #tf.reset_default_graph()
    print('test dataset: ', next_element, next_label)
    global_step, accuracy_mean, summary_op = build_model(next_element, next_label, False, keep_prob, lr, batch_size)
    saver = tf.train.Saver()
    eval_sess = tf.Session()
    print(eval_sess.run([tf.shape(next_element), tf.shape(next_label)]))
    eval_sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter('./graphs/convnettest', tf.get_default_graph())#train_sess.graph)
    path = tf.train.latest_checkpoint(checkpoints_path)
        
    if path:
        print('restore path: ', path)
        saver.restore(eval_sess,path)
        #print('global_step', global_step.eval(eval_sess))
    else:
        print('could not restore')
    total_acc = 0
    i = 0
    try:
        while True:
            accmean_, summary_, gs_ = eval_sess.run([accuracy_mean, summary_op, global_step])#,feed_dict={imgs_placeholder: x_train, labels_placeholder: y_train}) 
            print(accmean_)
            if (i%100==0):
                print(accmean_)
            writer.add_summary(summary_)
            total_acc += accmean_
            i += 1
    except tf.errors.OutOfRangeError:
        pass
    print('total_acc_test: ', total_acc/i)
    #writer.close()
    eval_sess.close()