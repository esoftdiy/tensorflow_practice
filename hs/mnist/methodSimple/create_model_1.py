# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST biginners tutorial by Google. 
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

"""A very simple MNIST classifier.
Documentation at
http://niektemme.com/ @@to do

This script is based on the Tensoflow MNIST beginners tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/beginners/index.html
"""

#import modules
import gzip
import os
import numpy as np
import tensorflow as tf
from tensorflow import global_variables_initializer
from tensorflow.examples.tutorials.mnist import input_data


#import data
mnist = input_data.read_data_sets("D:/Python/MNIST/data/", one_hot=True)

# Create the model
'''这里使用了一个28*28=784列的数据来表示一个图片的构成，也就是说，每一个点都是这个图片的一个特征，
这个其实比较好理解，因为每一个点都会对图片的样子和表达的含义有影响，只是影响的大小不同而已。
至于为什么要将28*28的矩阵摊平成为一个1行784列的一维数组，我猜测可能是因为这样做会更加简单直观。'''
x = tf.placeholder(tf.float32, shape=[None, 784])#数据形状。默认是None，就是一维值，也可以是多维（比如 [None, 3]表示列是3，行不定）x表示1行784列
W = tf.Variable(tf.zeros([784, 10]))#特征值对应的权重.W的矩阵为784行10列
b = tf.Variable(tf.zeros([10]))#偏置量 1行10列
y = tf.nn.softmax(tf.matmul(x, W) + b)#预测的结果矩阵 x*w+b=1行10列

# Define loss and optimizer
y_ = tf.placeholder(tf.float32, [None, 10])#真实结果
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init_op = global_variables_initializer()
saver = tf.train.Saver()


# Train the model and save the model to disk as a model.ckpt file
# file is stored in the same directory as this python script is started
"""
The use of 'with tf.Session() as sess:' is taken from the Tensor flow documentation
on on saving and restoring variables.
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
with tf.Session() as sess:
    sess.run(init_op)
    print(b)
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
        print("%s %d: %f"%("round",i,sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))
    save_path = saver.save(sess, "model.ckpt")
    print ("Model saved in file: ", save_path)


