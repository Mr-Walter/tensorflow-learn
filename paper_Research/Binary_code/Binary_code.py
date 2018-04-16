# -*- coding:utf-8 -*-

'''
cankao: https://github.com/technicolor-research/subic
https://github.com/kevinlin311tw/caffe-cvprw15
'''

import tensorflow as tf
import cv2
import numpy as np
from tensorflow.examples.tutorials.mnist.input_data import read_data_sets
from tensorflow.contrib import slim
import os
os.environ['CUDA_VISIBLE_DEVICES']='1' # 指定GPU from 0,1,2....

# Hyperparameter
lr=1e-4
batch_size=128
img_Pixels_h=28
img_Pixels_w=28
img_Pixels_c=1
num_classes=10
epochs=10
log_dir='./model'

if not os.path.exists(log_dir):os.mkdir(log_dir)


# load data
mnist=read_data_sets('./MNIST_data',one_hot=False)

class Net():
    def __init__(self,images,trainable=True,reuse=None):
        self.images=images
        self.img_Pixels_h=img_Pixels_h
        self.img_Pixels_w = img_Pixels_w
        self.img_Pixels_c = img_Pixels_c
        self.trainable=trainable
        self.reuse=reuse
        self.num_classes=num_classes

    def forward_Net(self):
        net=self.images # [n,28,28,1]

        net=tf.layers.conv2d(net,32,5,padding='same',
                             activation=tf.nn.relu,
                             trainable=self.trainable,
                             reuse=self.reuse,name='conv1') # [n,28,28,32]
        net=tf.layers.max_pooling2d(net,2,2,'same',name='pool1') # [n,14,14,32]


        net = tf.layers.conv2d(net, 64, 3, padding='same',
                               activation=tf.nn.relu,
                               trainable=self.trainable,
                               reuse=self.reuse, name='conv2')  # [n,14,14,64]
        net = tf.layers.max_pooling2d(net, 2, 2, 'same', name='pool2')  # [n,7,7,64]

        net = tf.reshape(net, [-1, np.int(np.prod(net.get_shape()[1:]))],name='flatten1') # [n,7*7*64]

        feature_layer=net # [n,7*7*64]

        net=tf.layers.dense(net,4096,activation=tf.nn.relu,trainable=self.trainable,reuse=self.reuse,name='fc1') # [n,4096]

        net = tf.layers.dense(net, self.num_classes, activation=tf.nn.softmax, trainable=self.trainable, reuse=self.reuse,
                              name='output')  # [n,10]

        return feature_layer,net


if __name__=='__main__':
    x=tf.placeholder(tf.float32,(None,img_Pixels_h,img_Pixels_w,img_Pixels_c))
    y=tf.placeholder(tf.int64,(None,))

    net=Net(x)
    _,pred=net.forward_Net()

    cost=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,logits=pred))
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    steps=mnist.train.num_examples // batch_size

    saver=tf.train.Saver(max_to_keep=1) # 最多保留一个版本
    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU70%的显存
    # session = tf.Session(config=config)

    with tf.Session(config=config) as sess:
        sess.run(init)
        for epoch in range(epochs):
            epoch_loss=0
            for step in range(steps):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                batch_x=np.reshape(batch_x,[-1,img_Pixels_h,img_Pixels_w,img_Pixels_c])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c

                if step%50==0:
                    print('epoch',epoch,'|','step:',step,'|','Train acc: ', accuracy.eval({x: batch_x, y: batch_y}),'|','loss:',c)

            print(epoch, ' : ', epoch_loss/steps)
            save_path=saver.save(sess,os.path.join(log_dir,'model.ckpt'),global_step=epoch)
            print('model save to ',save_path)
        # test
        batch_x = np.reshape(mnist.test.images, [-1, img_Pixels_h, img_Pixels_w, img_Pixels_c])
        print('准确率: ', accuracy.eval({x: batch_x, y: mnist.test.labels}))
