#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

'''
inception_resnet_v2  输入shape 299x299x3
inception_resnet_v2.py 下载地址：https://github.com/tensorflow/models/tree/master/research/slim/nets
inception_resnet_v2 model: https://github.com/tensorflow/models/tree/master/research/slim#Pretrained
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim
import inception_resnet_v2
import os
from tensorflow.python.platform import gfile
import argparse
import sys
import numpy as np
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

parse = argparse.ArgumentParser()
parse.add_argument('--mode', type=int, default=1, help='1 train ,0 valid')
parse.add_argument('-ta', '--trainable', type=bool, default=True, help='trainable or not')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parse.add_argument('--epochs', type=int, default=10, help='epochs')
arg = parse.parse_args()

# Hyperparameter
train = arg.mode  # 1 train ,0 test,2 输出编码测试
flag = False
lr = arg.lr  # 1e-3~1e-6
batch_size = 128  # 逐步改变 128~128*4
img_Pixels_h = 299
img_Pixels_w = 299
img_Pixels_c = 3
num_classes = 458
epochs = arg.epochs
log_dir = './model'
keep_rata = 0.7

first=True


if not os.path.exists(log_dir):os.mkdir(log_dir)


class Data():
    def __init__(self, data_path, batch_size):
        self.data_path = data_path
        self.batch_size = batch_size
        self.read()
        self.num_images = len(self.data)
        self.start = 0
        self.end = 0

    def read(self):
        data = []
        with open(self.data_path, 'r') as fp:
            for f in fp:
                data.append(f.strip('\n'))
        self.data = data

    def shuttle_data(self):
        '''打乱数据'''
        np.random.shuffle(self.data)
        self.start = 0
        self.end = 0

    def Next_batch(self):
        self.end = min((self.start + self.batch_size, self.num_images))
        data = self.data[self.start:self.end]

        self.start = self.end
        if self.start == self.num_images:
            self.start = 0

        images = []
        labels = []

        for da in data:
            da = da.strip('\n').split(',')
            labels.append(int(da[1]))
            images.append(cv2.imread(da[0]))

        # 特征归一化处理
        imgs = np.asarray(images, np.float32)
        imgs = (imgs - np.min(imgs, 0)) * 1. / (np.max(imgs, 0) - np.min(imgs, 0))
        # imgs = (imgs - np.mean(imgs, 0)) * 1. / np.std(imgs,0)

        return imgs, np.asarray(labels, np.int64)  # [batch_size,128,64,3],[batch_size,]


# # 数据增强
# https://blog.csdn.net/medium_hao/article/details/79227056
def distort_color(image, color_ordering=0):
    if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)  # 亮度
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)  # 饱和度
        image = tf.image.random_hue(image, max_delta=0.2)  # 色相
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)  # 对比度
    if color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    if color_ordering == 2:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
    if color_ordering == 3:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
    return tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image):
    # if bbox is None:
    #     bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # if image.dytpe != tf.float32:
    #    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # distorted_image = tf.image.resize_images(image, (height, width), method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    return distorted_image


with tf.Graph().as_default() as graph:
    # 加载数据
    if train == 1:
        data = Data('train.data', batch_size)
        # 训练时做数据增强，其他情况没必要做
        # 数据增强
        x = tf.placeholder(tf.float32, (batch_size, 128, 64, img_Pixels_c), name='x')
        x1 = []
        for i in range(x.shape[0]):
            # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
            x2 = tf.squeeze(tf.slice(x, [i, 0, 0, 0], [1, 128, 64, img_Pixels_c]), 0)
            x1.append(preprocess_for_train(x2))

        x1 = tf.convert_to_tensor(x1, tf.float32)
        x1 = tf.image.resize_image_with_crop_or_pad(x1, img_Pixels_h, img_Pixels_w)

    if train == 0:
        data = Data('valid.data', batch_size)
        x = tf.placeholder(tf.float32, (None, 128, 64, img_Pixels_c), name='x')
        x1 = tf.image.resize_image_with_crop_or_pad(x, img_Pixels_h, img_Pixels_w)


    y_ = tf.placeholder(tf.int64, [None,], 'y_')
    is_training=tf.placeholder(tf.bool, name='MODE')
    keep = tf.placeholder(tf.float32)

    with slim.arg_scope(inception_resnet_v2.inception_resnet_v2_arg_scope()):
        end_points= inception_resnet_v2.inception_resnet_v2(inputs=x1,num_classes=1001,is_training=is_training,dropout_keep_prob=keep)

    net=end_points[1]['PreLogitsFlatten']
    if not flag:
        net=tf.stop_gradient(net) # 这层与之前的层都不进行梯度更新

    print(net.shape)
    with tf.variable_scope('D'):
        fc1 = slim.fully_connected(net, 512, activation_fn=tf.nn.elu,trainable=arg.trainable,
                                      scope='fc1')
        fc = slim.fully_connected(fc1, 48, activation_fn=tf.nn.sigmoid,
                                   scope='coding_layer')
        y = slim.fully_connected(fc, num_classes, activation_fn=tf.nn.softmax,
                                      scope='output')

    # 只更新coding_layer与output这两层参数，其他的参数不更新
    tvars = tf.trainable_variables()  # 获取所有可以更新的变量
    d_params = [v for v in tvars if v.name.startswith('D/')]

    cost=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=y)

    if first:
        # 注 默认是更新所以参数 var_list=None
        train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost, var_list=d_params)
    else:
        train_op = tf.train.AdamOptimizer(lr).minimize(cost, var_list=d_params)

    # train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost,var_list=d_params)

    correct_prediction = tf.equal(tf.argmax(y,1), y_)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 占用GPU70%的显存

    sess=tf.InteractiveSession(config=config)
    sess.run(init)


    if first:
        # wget http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz
        # tar -zxf inception_resnet_v2_2016_08_30.tar.gz  -C output
        # 加载模型参数
        var_list = tf.global_variables()
        var_list_1 = []
        for var in var_list:  # 不加载 最后两层的参数，即重新训练
            if 'fc1' in var.name or 'coding_layer' in var.name or 'output' in var.name:
                # var_list_1.remove(var)
                continue
            var_list_1.append(var)
        var_list=None

        saver=tf.train.Saver(var_list=var_list_1)
        saver.restore(sess,'./output/inception_resnet_v2_2016_08_30.ckpt')
    else:
        saver = tf.train.Saver()
        # 验证之前是否已经保存了检查点文件
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)  # 只能导入参数

    steps = data.num_images // batch_size
    index2 = np.arange(0, batch_size)

    if train == 1:
        for epoch in range(epochs):
            data.shuttle_data()
            epoch_loss = 0
            for step in range(steps):
                batch_x, batch_y = data.Next_batch()

                # 每个step 打乱数据
                np.random.shuffle(index2)
                batch_x = batch_x[index2]
                batch_y = batch_y[index2]

                # feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False}
                feed_dict = {x: batch_x, is_training: True, y_: batch_y,keep:keep_rata}
                _, c = sess.run([train_op, cost], feed_dict)
                epoch_loss += c
                if step % 50 == 0:
                    # feed_dict = {images_placeholder: batch_x, phase_train_placeholder: False,y_true:batch_y}
                    acc = sess.run(accuracy, feed_dict)
                    print('epoch', epoch, 'step', step, '|', 'acc', acc, '|', 'loss', c)
            print(epoch, ' : ', epoch_loss / steps)

            # 保存所有变量 var_list指定要保持或者是提取的变量，默认是所有变量
            saver = tf.train.Saver(var_list=tf.global_variables())  # var_list=None 也是默认保持所以变量
            saver.save(sess, os.path.join(log_dir, 'model.ckpt'))

    sess.close()
