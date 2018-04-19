# -*- coding:utf-8 -*-

'''
DenseNet
参考：https://blog.csdn.net/u014380165/article/details/75142664
'''

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

parse=argparse.ArgumentParser()
parse.add_argument('--mode',type=int,default=1,help='1 train ,0 test,2 编码验证测试,3 编码测试')
parse.add_argument('-ta','--trainable',type=bool,default=True,help='trainable or not')
parse.add_argument('--lr',type=float,default=1e-4,help='learning rate')
parse.add_argument('--epochs',type=int,default=10,help='epochs')
arg=parse.parse_args()

# Hyperparameter
train = arg.mode  # 1 train ,0 test,2 输出编码测试
flag=arg.trainable
lr = arg.lr # 1e-3~1e-6
batch_size = 128 # 逐步改变 128~128*4
img_Pixels_h = 128
img_Pixels_w = 64
img_Pixels_c = 3
num_classes = 12
epochs = arg.epochs
log_dir = './model'
keep_rata=0.8

class Data():
    def __init__(self,data_path,batch_size):
        self.data_path=data_path
        self.batch_size=batch_size
        self.read()
        self.num_images=len(self.data)
        self.start = 0
        self.end = 0

    def read(self):
        data=[]
        with open(self.data_path,'r') as fp:
            for f in fp:
                data.append(f.strip('\n'))
        self.data=data

    def shuttle_data(self):
        '''打乱数据'''
        np.random.shuffle(self.data)
        self.start = 0
        self.end = 0

    def Next_batch(self):
        self.end=min((self.start+self.batch_size,self.num_images))
        data=self.data[self.start:self.end]

        self.start=self.end
        if self.start==self.num_images:
            self.start=0

        images=[]
        labels=[]

        for da in data:
            da=da.strip('\n').split(',')
            labels.append(int(da[1]))
            images.append(cv2.imread(da[0]))

        # 特征归一化处理
        imgs=np.asarray(images,np.float32)
        imgs=(imgs-np.min(imgs,0))*1./(np.max(imgs,0)-np.min(imgs,0))
        # imgs = (imgs - np.mean(imgs, 0)) * 1. / np.std(imgs,0)

        return imgs,np.asarray(labels,np.int64) # [batch_size,128,64,3],[batch_size,]

class Net():
    def __init__(self,images,dropout,trainable=True,reuse=None):
        self.images=images
        self.trainable=trainable
        self.reuse=reuse
        self.dropout=dropout
        self.forward_Net()

    def forward_Net(self):
        # net_work={}
        net=self.images # [n,128,64,3]
        # net=tf.layers.conv2d(net,64,7,(2,1),'same',kernel_initializer=tf.random_normal_initializer,
        #                      activation=tf.nn.leaky_relu,trainable=self.trainable,reuse=self.reuse,name='conv1') # [n,64,64,64]
        # net = slim.batch_norm(net)
        # net=slim.conv2d(net,num_outputs,7,(2,1),activation_fn=slim.relu,normalizer_fn=slim.layers.batch_norm,reuse=self.reuse,trainable=self.trainable,scope='conv1') # [n,64,64,32]

        net = slim.conv2d(net, 32, 7, (2,1),padding='SAME',activation_fn=tf.nn.leaky_relu,
                          normalizer_fn=slim.layers.batch_norm, reuse=self.reuse, trainable=self.trainable,
                          scope='conv1')  # [n,64,64,32]
        net=tf.layers.max_pooling2d(net,3,2,'same',name='pool1') # [n,32,32,32]

        # Dense Block-1
        branch_1=slim.conv2d(net,net.shape[-1]//2,1,1,padding='same',activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.layers.batch_norm,reuse=self.reuse,trainable=self.trainable,scope='db1_1_1')
        branch_1 = slim.conv2d(branch_1, net.shape[-1], 3, 1,padding='same',activation_fn=tf.nn.leaky_relu, normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                            trainable=self.trainable, scope='db1_1_2') # [n,32,32,32]

        net=tf.concat((net,branch_1),-1) # [n,32,32,64]

        branch_2 = slim.conv2d(net, net.shape[-1]//2, 1, 1, padding='same',activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_2_1') # [n,32,32,32]
        branch_2 = slim.conv2d(branch_2, net.shape[-1], 3, 1, padding='same',activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_2_2') # [n,32,32,64]

        net = tf.concat((net, branch_2), -1)  # [n,32,32,128]

        branch_3 = slim.conv2d(net, net.shape[-1]//2, 1, 1, padding='same',activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='db1_3_1')  # [n,32,32,64]
        branch_3 = slim.conv2d(branch_3, net.shape[-1], 3, 1, padding='same',activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.layers.batch_norm,
                               reuse=self.reuse,
                               trainable=self.trainable, scope='db1_3_2')  # [n,32,32,128]

        net = tf.concat((net, branch_3), -1)  # [n,32,32,256]

        # transition layer
        net = slim.conv2d(net, net.shape[-1]//2, 1, 1, padding='same',activation_fn=tf.nn.leaky_relu,normalizer_fn=slim.layers.batch_norm, reuse=self.reuse,
                               trainable=self.trainable, scope='tl1_1')  # [n,32,32,128]
        net=slim.avg_pool2d(net,2,2,'same',scope='pool2') # [n,16,16,128]

        net=tf.layers.dropout(net,self.dropout)

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

        net= slim.fully_connected(net,32,activation_fn=tf.nn.sigmoid,scope='coding_layer')
        self.net = net

        self.pred=slim.fully_connected(net,num_classes,activation_fn=tf.nn.softmax,scope='output')


if __name__=="__main__":
    data=Data('train.data',batch_size)
    # data.Next_batch()

    x = tf.placeholder(tf.float32, (None, img_Pixels_h, img_Pixels_w, img_Pixels_c))
    y = tf.placeholder(tf.int64, (None,))
    dropout=tf.placeholder(tf.float32)

    net=Net(x,dropout,flag)
    coding=net.net
    pred=net.pred

    # ------------------------
    # 计算lable到编码层的cost，label相同，对应的编码要相同,对应的编码cost最小，如果label不同，对应的编码cost最大，其倒数最小
    y1 = tf.slice(y, [0], [batch_size - 1])  # [n-1,]
    y2 = tf.slice(y, [1], [batch_size - 1])  # [n-1,]

    # label相同时
    error = tf.cast(tf.equal(y1, y2), tf.float32)  # [n-1,]
    # 编码
    net = tf.round(net.net)  # [n,32]

    net1 = tf.slice(net, [0, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]
    net2 = tf.slice(net, [1, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error
    net2 = net2 * error

    cost2 = tf.losses.mean_squared_error(net1, net2)

    # label不同时,cost最大，倒数最小(转成求倒数，因为最后要求的是整体cost 最小)
    error = tf.cast(tf.logical_not(tf.equal(y1, y2)), tf.float32)  # [n-1,]
    net1 = tf.slice(net, [0, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]
    net2 = tf.slice(net, [1, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error
    net2 = net2 * error

    cost3 = tf.losses.mean_squared_error(net1, net2)

    # -------------------------
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    cost = cost + cost2 + 1. / cost3  # cost最小 ，则cost3 最大

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 占用GPU70%的显存

    saver = tf.train.Saver(max_to_keep=1)

    steps =data.num_images // batch_size
    index2 = np.arange(0, batch_size)

    with tf.Session(config=config) as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if train == 1:
            for epoch in range(epochs):
                epoch_loss = 0
                # 每个epoch都打乱数据
                data.shuttle_data()

                for step in range(steps):
                    batch_x,batch_y=data.Next_batch()
                    # 每个step 打乱数据
                    np.random.shuffle(index2)
                    batch_x=batch_x[index2]
                    batch_y=batch_y[index2]

                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,dropout:keep_rata})
                    epoch_loss += c

                    if step % 50 == 0:
                        print('epoch', epoch, '|', 'step:', step, '|', 'Train acc: ',
                              accuracy.eval({x: batch_x, y: batch_y,dropout:keep_rata}), '|', 'loss:', c)

                print(epoch, ' : ', epoch_loss / steps)
                save_path = saver.save(sess, os.path.join(log_dir, 'model.ckpt'))  # , global_step=epoch)
                print('model save to ', save_path)

        if train==0:
            # test
            data = Data('train.data', batch_size*10)
            batch_x, batch_y = data.Next_batch()
            print('准确率: ', accuracy.eval({x: batch_x, y: batch_y,dropout:1.}))
        if train==2: # encoding 验证
            # Output encoding
            data = Data('train.data', batch_size * 10) # train.data
            batch_x, batch_y = data.Next_batch()
            feat = coding
            feat = tf.round(feat)
            feat = sess.run(feat, {x: batch_x,dropout:1.})  # [1000,48]
            # 转成对应的十进制数
            feat = feat.astype(np.int64)
            feat_Decimal = []  # 十进制数
            for a in feat:
                c = ''.join(str(a)[1:-1]).replace(' ', '').replace('\n', '')
                feat_Decimal.append(int(c, 2))

            labels = batch_y  # [10000,]

            # ---------------
            # label相同，编码相同
            labels = labels.astype(np.int32)
            error = labels[:-1] - labels[1:]
            er = (error == 0)
            print(np.sum(np.asarray(er,np.int16)))
            feat_Decimal = np.asarray(feat_Decimal, np.int64)
            print(np.mean(((feat_Decimal[:-1][er] - feat_Decimal[1:][er]) == 0).astype(np.int32)))  # 0.8627

            # label不同，编码相同
            er = (error != 0)
            print(np.mean(((feat_Decimal[:-1][er] - feat_Decimal[1:][er]) == 0).astype(np.int32)))

            # 编码相同，lable相同
            error = feat_Decimal[:-1] - feat_Decimal[1:]
            er = (error == 0)
            print(np.sum(np.asarray(er, np.int16)))
            print(np.mean(((labels[:-1][er] - labels[1:][er]) == 0).astype(np.int32)))  # 0.9959

            # 编码不同，label相同
            er = (error != 0)
            print(np.mean(((labels[:-1][er] - labels[1:][er]) == 0).astype(np.int32)))

        if train==3: # 图片做encoding
            data = Data('test.data', 20)
            batch_x, batch_y = data.Next_batch()
            feat = coding
            feat = tf.round(feat)
            feat = sess.run(feat, {x: batch_x, dropout: 1.})  # [1000,48]
            # 转成对应的十进制数
            feat = feat.astype(np.int64)
            feat_Decimal = []  # 十进制数
            for a in feat:
                c = ''.join(str(a)[1:-1]).replace(' ', '').replace('\n', '')
                feat_Decimal.append(int(c, 2))

            labels = batch_y  # [10000,]

            print('labels:',labels)
            print('encoding:',feat_Decimal)
