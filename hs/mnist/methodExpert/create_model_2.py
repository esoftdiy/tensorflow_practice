# Copyright 2016 Niek Temme.
# Adapted form the on the MNIST expert tutorial by Google. 
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

This script is based on the Tensoflow MNIST expert tutorial
See extensive documentation for the tutorial at
https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html
"""

#import modules
import tensorflow as tf
from tensorflow import global_variables_initializer
from tensorflow.examples.tutorials.mnist import input_data

#import data
mnist = input_data.read_data_sets("D:/Python/MNIST/data/", one_hot=True)

sess = tf.InteractiveSession()

# Create the model
# #创建两个占位符，x为输入网络的图像，y_为输入网络的图像类别
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)

fullConnectNerveCellNum=1024#全连接层神经元个数
#权重初始化函数
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)#输出服从截尾正态分布的随机值
  return tf.Variable(initial)

#偏置初始化函数
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

#创建卷积op
#x 是一个4维张量，shape为[batch,height,width,channels]
#卷积核移动步长为1。填充类型为SAME,可以不丢弃任何像素点, VALID丢弃边缘像素点
def conv2d(x, filter):
  return tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')

#创建池化op
#采用最大池化，也就是取窗口中的最大值作为结果
#x 是一个4维张量，shape为[batch,height,width,channels]
#ksize表示pool窗口大小为2x2,也就是高2，宽2
#strides，表示在height和width维度上的步长都为2
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

#第1层，卷积层
#初始化filter为[5,5,1,32]的张量，表示卷积核大小为5*5，1表示图像通道数，32表示卷积核个数即输出32个特征图
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

#把输入x(二维张量,shape为[batch, 784])变成4d的x_image，x_image的shape应该是[batch,28,28,1]
#-1表示自动推测这个维度的size
x_image = tf.reshape(x, [-1,28,28,1])
#把x_image和权重进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max_pooling
#h_pool1的输出即为第一层网络输出，shape为[batch,14,14,32]
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#第2层，卷积层
#卷积核大小依然是5*5，通道数为32，卷积核个数为64
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, fullConnectNerveCellNum])
b_fc1 = bias_variable([fullConnectNerveCellNum])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([fullConnectNerveCellNum, 10])
b_fc2 = bias_variable([10])

y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Define loss and optimizer
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


"""
Train the model and save the model to disk as a model2.ckpt file
file is stored in the same directory as this python script is started

Based on the documentatoin at
https://www.tensorflow.org/versions/master/how_tos/variables/index.html
"""
saver = tf.train.Saver()
sess.run(global_variables_initializer())
#with tf.Session() as sess:
    #sess.run(init_op)
for i in range(5000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y_: batch[1], keep_prob: 1.0})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

save_path = saver.save(sess, "model2.ckpt")
print ("Model saved in file: ", save_path)

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))



