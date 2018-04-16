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

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

# Hyperparameter
train = 0  # 1 classify ,0 Binary code train ,-1 输出编码测试
lr = 1e-4 # 1e-3~1e-6
batch_size = 128 # 逐步改变 128~128*4
img_Pixels_h = 28
img_Pixels_w = 28
img_Pixels_c = 1
num_classes = 10
epochs = 10
log_dir = './model'

if not os.path.exists(log_dir): os.mkdir(log_dir)

# load data
mnist = read_data_sets('./MNIST_data', one_hot=False)


def forward_Net(images, trainable=True, reuse=None):
    net = images  # [n,28,28,1]

    net = tf.layers.conv2d(net, 32, 5, padding='same',
                           activation=tf.nn.relu,
                           trainable=trainable,
                           reuse=reuse, name='conv1')  # [n,28,28,32]
    net = tf.layers.max_pooling2d(net, 2, 2, 'same', name='pool1')  # [n,14,14,32]

    net = tf.layers.conv2d(net, 64, 3, padding='same',
                           activation=tf.nn.relu,
                           trainable=trainable,
                           reuse=reuse, name='conv2')  # [n,14,14,64]
    net = tf.layers.max_pooling2d(net, 2, 2, 'same', name='pool2')  # [n,7,7,64]

    net = tf.reshape(net, [-1, np.int(np.prod(net.get_shape()[1:]))], name='flatten1')  # [n,7*7*64]

    feature_layer = net  # [n,7*7*64]

    net = tf.layers.dense(net, 4096, activation=tf.nn.relu,
                          name='fc1')  # [n,4096]

    net = tf.layers.dense(net, num_classes, activation=tf.nn.softmax,
                          name='output')  # [n,10]

    return feature_layer, net


if __name__ == '__main__':
    x = tf.placeholder(tf.float32, (None, img_Pixels_h, img_Pixels_w, img_Pixels_c))
    y = tf.placeholder(tf.int64, (None,))

    # Classify
    if train == 1:
        _, pred = forward_Net(x)

        # tf.global_variables()
        saver = tf.train.Saver(max_to_keep=1)  # 最多保留一个版本
        '''
        var_list = tf.global_variables()
        print(var_list)
        var_list_1=[]
        for var in var_list:
            if 'fc1' in var.name  or 'ouput' in var.name:
                # var_list_1.remove(var)
                continue
            var_list_1.append(var)
        print(var_list_1)
        '''
    # Binary code
    if train == 0 or train == -1:
        feature, _ = forward_Net(x, False, tf.AUTO_REUSE)  # 前几次训练trainable设为False，后面微调参数时可以设置为True
        # 连接一个隐藏层作为特征编码
        net = tf.layers.dense(feature, 4096, activation=tf.nn.relu,
                              name='fc2')  # [n,4096]
        net = tf.layers.dense(net, 32, activation=tf.nn.sigmoid, # 32,16,8
                              name='fc3')  # [n,48]  # Binary code layer
        # net=tf.round(net) # 第一次训练不加，第二次加上

        pred = tf.layers.dense(net, num_classes, activation=tf.nn.softmax, name='output2')

    if train == 0:
        '''
        第一次是从classify模型提取参数（需要排除掉没有的变量），下一次是从Binary code模型提取参数（不需要做变量排除）
        '''
        # saver = tf.train.Saver(max_to_keep=1)
        # '''
        var_list = tf.global_variables()  # 得到所有变量
        # 删除掉这些scope为fc2，fc3，output2的所有变量,后续模型参数保存时的全加上
        var_list_1 = []
        for var in var_list:
            if 'fc2' in var.name or 'fc3' in var.name or 'output2' in var.name:
                # var_list.remove(var)
                continue
            var_list_1.append(var)
        # print(var_list_1)
        saver = tf.train.Saver(max_to_keep=1, var_list=var_list_1)  # 最多保留一个版本
        # '''
        '''
        注:需要删除掉这些scope为fc2，fc3，output2的所有变量，因为classify训练的参数没有这些变量直接提起变量会报错
        saver 默认提取所有变量 var_list=tf.global_variables() # 得到所有变量
        '''
    if train == -1:
        saver = tf.train.Saver(max_to_keep=1)

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))

    if train==0:
        # 计算lable到编码层的cost，label相同，对应的编码要相同
        y1 = tf.slice(y, [0], [batch_size - 1]) # [n-1,]
        y2 = tf.slice(y, [1], [batch_size - 1]) # [n-1,]
        error = tf.cast(tf.equal(y1, y2), tf.float32) # [n-1,]
        # 编码
        net=tf.round(net) # [n,32]

        net1=tf.slice(net, [0,0], [batch_size - 1,net.shape[-1]]) # [n-1,32]
        net2 = tf.slice(net, [1,0], [batch_size - 1, net.shape[-1]])  # [n-1,32]

        error=tf.expand_dims(error,-1) # [n-1,1]
        net1=net1*error
        net2=net2*error

        cost2 = tf.losses.mean_squared_error(net1, net2)

        # cost=abs(cost)+abs(cost2)
        cost = cost + cost2

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()

    steps = mnist.train.num_examples // batch_size

    # Launch the graph
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU70%的显存
    # session = tf.Session(config=config)

    with tf.Session(config=config) as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(log_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        if train == 0:
            saver = tf.train.Saver(max_to_keep=1)  # 模型参数保存时保存所有变量
        if train != -1:
            for epoch in range(epochs):
                epoch_loss = 0
                for step in range(steps):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    batch_x = np.reshape(batch_x, [-1, img_Pixels_h, img_Pixels_w, img_Pixels_c])
                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                    epoch_loss += c

                    if step % 50 == 0:
                        print('epoch', epoch, '|', 'step:', step, '|', 'Train acc: ',
                              accuracy.eval({x: batch_x, y: batch_y}), '|', 'loss:', c)

                print(epoch, ' : ', epoch_loss / steps)
                save_path = saver.save(sess, os.path.join(log_dir, 'model.ckpt'))  # , global_step=epoch)
                print('model save to ', save_path)
                # test
            batch_x = np.reshape(mnist.test.images, [-1, img_Pixels_h, img_Pixels_w, img_Pixels_c])
            print('准确率: ', accuracy.eval({x: batch_x, y: mnist.test.labels}))

        if train == -1:
            # Output encoding
            feat = net
            feat = tf.round(feat)
            feat = sess.run(feat, {
                x: np.reshape(mnist.test.images, [-1, img_Pixels_h, img_Pixels_w, img_Pixels_c])})  # [1000,48]
            # 转成对应的十进制数
            feat = feat.astype(np.int64)
            feat_Decimal = []  # 十进制数
            for a in feat:
                c = ''.join(str(a)[1:-1]).replace(' ', '').replace('\n', '')
                feat_Decimal.append(int(c, 2))

            labels = mnist.test.labels  # [10000,]

            '''
            # label相同编码是否一致
            error=labels[:-1]-labels[1:]
            labels_same=np.sum((error==0).astype(np.int32))
            print(labels_same)
            labels_=labels+np.asarray(feat_Decimal,np.int64)
            error_ = labels_[:-1] - labels_[1:]
            labels_same_ = np.sum((error_ == 0).astype(np.int32))
            print(labels_same_)
            print('acc:',1.*labels_same_/labels_same) # acc: 0.7337278106508875
            # 编码相同lable是否相同
            labels_ = np.asarray(feat_Decimal, np.int64)
            error_ = labels_[:-1] - labels_[1:]
            labels_same_ = np.sum((error_ == 0).astype(np.int32))
            print(labels_same_)
            '''
            # ---------------
            # label相同编码是否一致
            labels = labels.astype(np.int32)
            error = labels[:-1] - labels[1:]
            er = (error == 0)
            feat_Decimal = np.asarray(feat_Decimal, np.int64)
            print(np.mean(((feat_Decimal[:-1][er] - feat_Decimal[1:][er]) == 0).astype(np.int32)))  # 0.8627

            # 编码相同lable是否相同
            error = feat_Decimal[:-1] - feat_Decimal[1:]
            er = (error == 0)
            print(np.mean(((labels[:-1][er] - labels[1:][er]) == 0).astype(np.int32))) # 0.9959
