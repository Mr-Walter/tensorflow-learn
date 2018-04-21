# -*- coding:utf-8 -*-

'''
DenseNet
参考：https://blog.csdn.net/u014380165/article/details/75142664

跨模态检索的自监督对抗哈希网络
参考：https://arxiv.org/pdf/1804.01223.pdf
'''

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
import os
import argparse
import pandas as pd

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

parse = argparse.ArgumentParser()
parse.add_argument('--mode', type=int, default=1, help='1 train ,0 valid')
parse.add_argument('-ta', '--trainable', type=bool, default=True, help='trainable or not')
parse.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parse.add_argument('--epochs', type=int, default=10, help='epochs')
arg = parse.parse_args()

# Hyperparameter
train = arg.mode  # 1 train ,0 test,2 输出编码测试
flag = arg.trainable
lr = arg.lr  # 1e-3~1e-6
batch_size = 64  # 逐步改变 128~128*4
img_Pixels_h = 128
img_Pixels_w = 64
img_Pixels_c = 3
num_classes = 10
epochs = arg.epochs
log_dir = './model'
keep_rata = 0.7

# 编码列数
m=32


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

def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # if image.dytpe != tf.float32:
    #    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)
    # distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    return distorted_image


class Net():
    def __init__(self, images,labels,dropout, trainable=True, reuse=None):
        self.images = images
        self.labels=labels
        self.trainable = trainable
        self.reuse = reuse
        self.dropout = dropout
        self.LabNet()
        self.ImgNet()

    def ImgNet(self):
        net = self.images  # [n,128,64,3]

        if train==1: # 训练时做数据增强，其他情况没必要做
            # 数据增强
            x1 = []
            for i in range(net.shape[0]):
                # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
                x2 = tf.squeeze(tf.slice(net, [i, 0, 0, 0], [1, img_Pixels_h, img_Pixels_w, img_Pixels_c]), 0)
                x1.append(preprocess_for_train(x2, img_Pixels_h, img_Pixels_w, None))

            net = tf.convert_to_tensor(x1, tf.float32)

        net = slim.conv2d(net, 32, 7, (2, 1), padding='SAME', activation_fn=tf.nn.leaky_relu,
                          normalizer_fn=slim.layers.batch_norm, reuse=self.reuse, trainable=self.trainable,
                          scope='conv1')  # [n,64,64,32]
        net = tf.layers.max_pooling2d(net, 3, 2, 'same', name='pool1')  # [n,32,32,32]
        # Dense Block-1
        branch_1 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse, trainable=self.trainable,
                               scope='db1_1_1')
        branch_1 = slim.conv2d(branch_1, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_1_2')  # [n,32,32,32]

        net = tf.concat((net, branch_1), -1)  # [n,32,32,64]

        branch_2 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_2_1')  # [n,32,32,32]
        branch_2 = slim.conv2d(branch_2, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_2_2')  # [n,32,32,64]

        net = tf.concat((net, branch_2), -1)  # [n,32,32,128]

        branch_3 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_3_1')  # [n,32,32,64]
        branch_3 = slim.conv2d(branch_3, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm,
                               reuse=self.reuse,
                               trainable=self.trainable, scope='db1_3_2')  # [n,32,32,128]

        net = tf.concat((net, branch_3), -1)  # [n,32,32,256]

        # transition layer
        net = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                          normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                          trainable=self.trainable, scope='tl1_1')  # [n,32,32,128]
        net = slim.avg_pool2d(net, 2, 2, 'same', scope='pool2')  # [n,16,16,128]

        net = tf.layers.dropout(net, self.dropout)

        # Dense Block-2
        branch_1 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse, trainable=self.trainable,
                               scope='db2_1_1')
        branch_1 = slim.conv2d(branch_1, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db2_1_2')  # [n,16,16,128]

        net = tf.concat((net, branch_1), -1)  # [n,32,32,256]

        branch_2 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db2_2_1')  # [n,16,16,128]
        branch_2 = slim.conv2d(branch_2, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db2_2_2')  # [n,16,16,256]

        net = tf.concat((net, branch_2), -1)  # [n,16,16,512]

        branch_3 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db2_3_1')  # [n,16,16,256]
        branch_3 = slim.conv2d(branch_3, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm,
                               reuse=self.reuse,
                               trainable=self.trainable, scope='db2_3_2')  # [n,16,16,512]

        net = tf.concat((net, branch_3), -1)  # [n,16,16,1024]

        # transition layer
        net = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                          normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                          trainable=self.trainable, scope='tl2_1')  # [n,16,16,512]
        net = slim.avg_pool2d(net, 2, 2, 'same', scope='pool2')  # [n,8,8,512]

        net = tf.layers.dropout(net, self.dropout)

        # Dense Block-3
        branch_1 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse, trainable=self.trainable,
                               scope='db3_1_1')
        branch_1 = slim.conv2d(branch_1, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db3_1_2')  # [n,16,16,128]

        net = tf.concat((net, branch_1), -1)  # [n,32,32,256]

        branch_2 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db3_2_1')  # [n,16,16,128]
        branch_2 = slim.conv2d(branch_2, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db3_2_2')  # [n,16,16,256]

        net = tf.concat((net, branch_2), -1)  # [n,16,16,512]

        branch_3 = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db3_3_1')  # [n,16,16,256]
        branch_3 = slim.conv2d(branch_3, net.shape[-1], 3, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                               normalizer_fn=slim.layers.batch_norm,
                               reuse=self.reuse,
                               trainable=self.trainable, scope='db3_3_2')  # [n,16,16,512]

        net = tf.concat((net, branch_3), -1)  # [n,16,16,1024]

        # transition layer
        net = slim.conv2d(net, net.shape[-1] // 2, 1, 1, padding='same', activation_fn=tf.nn.leaky_relu,
                          normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                          trainable=self.trainable, scope='tl3_1')  # [n,16,16,512]
        net = slim.avg_pool2d(net, 2, 2, 'same', scope='pool3')  # [n,4,4,512]

        net = tf.layers.dropout(net, self.dropout)

        # Classifications
        net = slim.avg_pool2d(net, 4, 1, 'valid', scope='pool2')  # [n,1,1,512]

        net = tf.reshape(net, [-1, np.int(np.prod(net.get_shape()[1:]))], name='flatten1')  # [n,512]

        net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu,trainable=self.trainable, scope='fc1')

        self.ImgNet_fc=net

        net = slim.fully_connected(net,m, activation_fn=None, scope='coding_layer') # sigmoid ,softmax

        self.ImgNet_label = tf.nn.sigmoid(net, name='ImgNet_label')
        # self.ImgNet_hash = tf.nn.softmax(net, name='ImgNet_hash')
        # self.ImgNet_hash_2 = tf.nn.tanh(net, name='ImgNet_hash_2')  # {-1,1}

    # 将label映射到哈希编码空间
    def LabNet(self):
        '''
        将label 转成类似one_hot编码
        （由于一张图片只有一个对象（label里只有一个1），此时的编码就是one_hot编码）
        如果图片里有多个对象，label里有对应个1
        （L→4096→512→N）
        :return: 哈希空间下的label编码
        '''
        if np.ndim(self.labels)<2:
            net=tf.one_hot(self.labels,num_classes) # [n,num_classes]
        net=tf.layers.dense(net,4096,activation=tf.nn.relu,trainable=self.trainable,reuse=self.reuse,name='labnet_fc1')
        net=tf.layers.dense(net,512,activation=tf.nn.relu,trainable=self.trainable,reuse=self.reuse,name='labnet_fc2')

        self.LabNet_fc = net
        net = tf.layers.dense(net, m, activation=None,name='labnet_hash')

        self.LabNet_label=tf.nn.sigmoid(net,name='labnet_label')
        # self.LabNet_hash=tf.nn.softmax(net,name='labnet_hash')
        # self.LabNet_hash_2 = tf.nn.tanh(net, name='labnet_hash_2') # {-1,1}

        # ----------重构label-------------------
        net=tf.nn.relu(net)
        net=tf.layers.dense(net,512,activation=tf.nn.relu,name='labnet_fc3')
        net = tf.layers.dense(net, 4096, activation=tf.nn.relu, name='labnet_fc4')
        net = tf.layers.dense(net, num_classes, activation=tf.nn.softmax, name='output')
        self.LabNet_new_label=net


if __name__ == "__main__":
    data = Data('train.data', batch_size)

    if train==1:
        x = tf.placeholder(tf.float32, (batch_size, img_Pixels_h, img_Pixels_w, img_Pixels_c))
    else:
        x = tf.placeholder(tf.float32, (None, img_Pixels_h, img_Pixels_w, img_Pixels_c))
    y = tf.placeholder(tf.int64, (None,))
    dropout = tf.placeholder(tf.float32)

    net=Net(x,y,dropout,flag)
    LabNet_fc=net.LabNet_fc
    LabNet_label = net.LabNet_label
    # LabNet_hash = tf.sign(net.LabNet_hash) # 转成 0、1编码
    # LabNet_hash = net.LabNet_hash
    # LabNet_hash_2=net.LabNet_hash_2
    LabNet_label_2=tf.round(LabNet_label)


    ImgNet_fc=net.ImgNet_fc
    ImgNet_label=net.ImgNet_label
    # ImgNet_hash=tf.sign(net.ImgNet_hash)
    # ImgNet_hash = net.ImgNet_hash
    # ImgNet_hash_2=net.LabNet_hash_2
    ImgNet_label_2 = tf.round(ImgNet_label)

    # cost
    # cost1=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=LabNet_fc,logits=ImgNet_fc))
    cost1=tf.losses.mean_squared_error(labels=LabNet_fc,predictions=ImgNet_fc)
    # cost2=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=LabNet_hash,logits=ImgNet_hash))
    cost3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=LabNet_label, logits=ImgNet_label))

    # cost4=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=LabNet_hash_2,logits=ImgNet_hash_2))

    cost5=m-tf.reduce_mean(tf.reduce_sum(tf.cast(tf.equal(LabNet_label_2,ImgNet_label_2),tf.float32)))

    cost4=tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=net.LabNet_new_label))

    # -----------------------

    # 计算lable到编码层的cost，label相同，对应的编码要相同,对应的编码cost最小，如果label不同，对应的编码cost最大，其倒数最小
    y1 = tf.slice(y, [0], [batch_size - 1])  # [n-1,]
    y2 = tf.slice(y, [1], [batch_size - 1])  # [n-1,]

    # label相同时
    error = tf.cast(tf.equal(y1, y2), tf.float32)  # [n-1,]
    # 编码
    # net = tf.round(net.net)  # [n,32]
    label = LabNet_label
    net1 = tf.slice(label, [0, 0], [batch_size - 1, label.shape[-1]])  # [n-1,32]
    net2 = tf.slice(label, [1, 0], [batch_size - 1, label.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error  # [n-1,m*k]
    net2 = net2 * error  # [n-1,m*k]

    cost6 = tf.losses.mean_squared_error(net1, net2)

    # label不同时,cost最大，倒数最小(转成求倒数，因为最后要求的是整体cost 最小)
    error = tf.cast(tf.logical_not(tf.equal(y1, y2)), tf.float32)  # [n-1,]
    net1 = tf.slice(label, [0, 0], [batch_size - 1, label.shape[-1]])  # [n-1,32]
    net2 = tf.slice(label, [1, 0], [batch_size - 1, label.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error
    net2 = net2 * error

    cost7 = tf.losses.mean_squared_error(net1, net2)

    # -------------------------

    cost = cost1 + cost3+cost5+cost6-cost7*10 +cost4
    # cost=cost1+cost2+cost3+cost4

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    # correct_pred = tf.equal(tf.argmax(pred, 1), y)
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU70%的显存

    saver = tf.train.Saver(max_to_keep=1)

    steps = data.num_images // batch_size
    index2 = np.arange(0, batch_size)

    with tf.Session(config=config) as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 训练
        if train == 1:
            for epoch in range(epochs):
                epoch_loss = 0
                # 每个epoch都打乱数据
                data.shuttle_data()

                for step in range(steps):
                    batch_x, batch_y = data.Next_batch()
                    # 每个step 打乱数据
                    np.random.shuffle(index2)
                    batch_x = batch_x[index2]
                    batch_y = batch_y[index2]

                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y, dropout: keep_rata})
                    epoch_loss += c

                    if step % 50 == 0:
                        print('epoch', epoch, '|', 'step:', step, '|', 'loss:', c)

                        # print(sess.run(tf.round(ImgNet_label),{x: batch_x, y: batch_y, dropout: keep_rata})[:5])
                        # exit(-1)

                print(epoch, ' : ', epoch_loss / steps)
                save_path = saver.save(sess, os.path.join(log_dir, 'model.ckpt'))  # , global_step=epoch)
                print('model save to ', save_path)

        if train==0:
            data = Data('valid.data', batch_size)  # train.data
            data.shuttle_data()
            same_hash_0 = []
            same_hash_1 = []
            same_hash_2 = []
            same_hash_3 = []
            same_hash_4 = []
            same_hash_5 = []
            same_hash_6 = []
            same_hash_7 = []
            same_hash_8 = []
            same_hash_9 = []
            same_hash=[same_hash_0,same_hash_1,same_hash_2,same_hash_3,same_hash_4,
                       same_hash_5,same_hash_6,same_hash_7,same_hash_8,same_hash_9]

            for step in range(steps):
                batch_x, batch_y = data.Next_batch()
                # print(batch_y[:10])
                # print(sess.run(tf.round(ImgNet_label), {x: batch_x, y: batch_y, dropout: keep_rata})[:10])
                # exit(-1)
                hash_code=sess.run(tf.round(ImgNet_label), {x: batch_x, dropout: 1.})
                for i in range(len(batch_y)):
                    for j in range(10):
                        if batch_y[i]==j:
                            same_hash[j].append(hash_code[i])

            for j in range(10):
                with open('hash_'+str(j)+'.data','w') as fp:
                    pd.DataFrame(np.asarray(same_hash[j],np.uint8)).to_csv(fp,index=False,header=False)




