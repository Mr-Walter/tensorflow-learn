# 计算真实标签与编码之间的cost

import tensorflow as tf
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
import numpy as np

mnist=read_data_sets('./MNIST_data',one_hot=False)

y=tf.placeholder(tf.int64,[None,])
batch_size=50
y1=tf.slice(y,[0],[batch_size-1])
y2=tf.slice(y,[1],[batch_size-1])

error=tf.cast(tf.equal(y1,y2),tf.float32)

y1=tf.to_float(y1)*error
y2=tf.to_float(y2)*error

cost=tf.losses.mean_squared_error(y1,y2)
batch_y=mnist.train.labels[:batch_size]

sess=tf.InteractiveSession()
tf.global_variables_initializer().run()
print(sess.run(error,{y:batch_y}))
print(sess.run(y1,{y:batch_y}))
print(sess.run(y2,{y:batch_y}))
print(sess.run(cost,{y:batch_y}))
sess.close()
