# -*- coding:utf-8 -*-

'''
DenseNet
参考：https://blog.csdn.net/u014380165/article/details/75142664

sigmoid 编码 round取整 [0,1,1,0,1] #有多个1
softmax 编码 最大值只会有一个 argmax [0,0,1,0,0] #只有一个1 可以直接用一个索引值代替 如：这里是2

如果最后编码层 shape [1,m*256]
使用sigmoid 编码 为[1,m*256] # {0,1}^(m*256)

使用softmax 编码 为[1,m*256]-->[1,m,256]-->[m,256]
每一个[1,256]对应一个索引值 --->[m,1] 如[8,9,10,……]其中数字为每一行的索引值

'''

import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np
import cv2
import os
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

parse = argparse.ArgumentParser()
parse.add_argument('--mode', type=int, default=-1, help='1 train ,0 test,2 编码验证测试,3 编码测试')
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

m = 8
k = 4 #256


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
    def __init__(self, images, dropout, trainable=True, reuse=None):
        self.images = images
        self.trainable = trainable
        self.reuse = reuse
        self.dropout = dropout
        self.forward_Net()

    def forward_Net(self):
        # net_work={}
        net = self.images  # [n,128,64,3]

        x1 = []
        # x2=[]
        for i in range(net.shape[0]):
            # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
            x2 = tf.squeeze(tf.slice(net, [i, 0, 0, 0], [1, img_Pixels_h, img_Pixels_w, img_Pixels_c]), 0)
            x1.append(preprocess_for_train(x2, img_Pixels_h, img_Pixels_w, None))

            # 数据增强
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

        self.fc7 = net
        # net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu,trainable=self.trainable, scope='fc1')

        net = slim.fully_connected(net, m * k, activation_fn=tf.nn.sigmoid, scope='coding_layer')

        self.net = net  # [n,m*k]

        self.pred = slim.fully_connected(net, num_classes, activation_fn=tf.nn.softmax, scope='output')


if __name__ == "__main__":
    data = Data('train.data', batch_size)
    # data.Next_batch()

    x = tf.placeholder(tf.float32, (batch_size, img_Pixels_h, img_Pixels_w, img_Pixels_c))
    y = tf.placeholder(tf.int64, (None,))
    dropout = tf.placeholder(tf.float32)

    net = Net(x, dropout, flag)
    coding = net.net
    pred = net.pred
    fc7 = net.fc7

    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred)) / np.log(
        num_classes)

    # 计算lable到编码层的cost，label相同，对应的编码要相同,对应的编码cost最小，如果label不同，对应的编码cost最大，其倒数最小
    y1 = tf.slice(y, [0], [batch_size - 1])  # [n-1,]
    y2 = tf.slice(y, [1], [batch_size - 1])  # [n-1,]

    # label相同时
    error = tf.cast(tf.equal(y1, y2), tf.float32)  # [n-1,]
    # 编码
    # net = tf.round(net.net)  # [n,32]
    net = net.net
    net1 = tf.slice(net, [0, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]
    net2 = tf.slice(net, [1, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error  # [n-1,m*k]
    net2 = net2 * error  # [n-1,m*k]

    cost2 = tf.losses.mean_squared_error(net1, net2) / (m * k)

    # label不同时,cost最大，倒数最小(转成求倒数，因为最后要求的是整体cost 最小)
    error = tf.cast(tf.logical_not(tf.equal(y1, y2)), tf.float32)  # [n-1,]
    net1 = tf.slice(net, [0, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]
    net2 = tf.slice(net, [1, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error
    net2 = net2 * error

    cost3 = tf.losses.mean_squared_error(net1, net2) / (m * k)

    # 先训练cost（不包含cost2、cost3），再冻结编码层以前的所有层，使用下面的cost 微调编码层
    cost = (8 * cost + 3 * cost2 - cost3) / 10  # cost最小 ，则cost3 最大

    # cost = cost + 1. * ind_entropy - 1. * batch_entropy # 总loss

    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

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
                        print('epoch', epoch, '|', 'step:', step, '|', 'Train acc: ',
                              accuracy.eval({x: batch_x, y: batch_y, dropout: keep_rata}), '|', 'loss:', c)

                print(epoch, ' : ', epoch_loss / steps)
                save_path = saver.save(sess, os.path.join(log_dir, 'model.ckpt'))  # , global_step=epoch)
                print('model save to ', save_path)

        # 验证
        if train == 0:
            # 每个类别计算ap，所有类别取平均得到mAp
            def voc_ap(rec, prec, use_07_metric=False):
                """Compute VOC AP given precision and recall. If use_07_metric is true, uses
                the VOC 07 11-point method (default:False).
                """
                if use_07_metric:
                    # 11 point metric
                    ap = 0.
                    for t in np.arange(0., 1.1, 0.1):
                        if np.sum(rec >= t) == 0:
                            p = 0
                        else:
                            p = np.max(prec[rec >= t])
                        ap = ap + p / 11.
                else:
                    # correct AP calculation
                    # first append sentinel values at the end
                    mrec = np.concatenate(([0.], rec, [1.]))
                    mpre = np.concatenate(([0.], prec, [0.]))

                    # compute the precision envelope
                    for i in range(mpre.size - 1, 0, -1):
                        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                    # to calculate area under PR curve, look for points
                    # where X axis (recall) changes value
                    i = np.where(mrec[1:] != mrec[:-1])[0]

                    # and sum (\Delta recall) * prec
                    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                return ap


            data = Data('valid.data', batch_size)  # train.data
            data.shuttle_data()

            N = data.num_images
            feat = coding
            steps = N // batch_size
            # N=steps*batch_size
            # 样本较多，按批进行
            encoding = np.zeros((N, m * k))  # 记录每张图片的编码  [64,2048]
            labels = np.zeros((N,), np.int64)  # 记录每张图的真实标签 [64,]
            score = np.zeros((N, num_classes))  # 记录每张图片的分成每一类的概率值  [64,10]
            # cols=c7.shape[-1]
            cols = fc7.get_shape().as_list()[-1]
            cols = int(cols)
            fc7_Eu_ = np.zeros((N, cols))  # [64,2048]

            for step in range(steps):
                batch_x, batch_y = data.Next_batch()

                feat_1 = sess.run(feat, {x: batch_x, dropout: 1.})  # 编码
                pred_1 = sess.run(pred, {x: batch_x, dropout: 1.})  # 预测类别
                fc7_1 = sess.run(fc7, {x: batch_x, dropout: 1.})
                encoding[step * batch_size:(step + 1) * batch_size] = feat_1
                score[step * batch_size:(step + 1) * batch_size] = pred_1
                labels[step * batch_size:(step + 1) * batch_size] = batch_y
                fc7_Eu_[step * batch_size:(step + 1) * batch_size] = fc7_1

            # print(encoding.shape)
            # print(labels.shape)
            # print(score.shape)
            # print(fc7.shape)
            # exit(-1)
            # 假设第一张图为查询图片，记录其他图片分成该类别的概率，与该图编码一致记为1,否则为0,与该图的真实类相同记为1,否则为0;

            score_1 = score[1:N, int(labels[0])]  # [N-1,] [63,]

            encoding0 = encoding.copy()

            # 根据编码粗搜索
            encoding = np.round(encoding).astype(np.uint8)  # [N,m*k]

            encoding_1 = np.bitwise_xor(encoding[1:], encoding[0])

            encoding_2 = np.asarray(np.sum(encoding_1, 1) < int(m * k * 2 * 0.18), np.uint8)  # [n-1,] 0.18～0.2

            encoding_8=np.sum(encoding_1, 1)

            encoding_2_1 = encoding_2.copy()

            # 精搜索
            '''
            # 取编码层的上一层做欧式距离

            fc8 = []
            index = []
            for i, x in enumerate(encoding_2):
                if x:
                    fc8.append(fc7_Eu_[1:][i])
                    index.append(i)

            # fc8与fc7[0]做欧式距离，fc8为0的一行跳过
            fc8=np.asarray(fc8)
            if len(fc8)==0:exit(-1)

            error=fc8-fc7_Eu_[0]

            error=error**2

            Eu_dis=np.sum(error,1)/cols
            print('Eu_dis', Eu_dis.shape)
            # Eu_dis=np.sum((np.squeeze(fc8,1)-fc7[0])**2,1)#/cols

            Eu_dis=np.asarray((Eu_dis < 0.16), np.uint8) # [x,] 0.1～0.25
            print('Eu_dis', Eu_dis)
            # 结合欧式距离，返回到源图的序号上
            encoding_3=encoding_2.copy()
            '''
            # ---------------------------------------------------------

            # 使用编码层求欧式距离
            fc8 = []
            index = []
            for i, x in enumerate(encoding_2):
                if x:
                    fc8.append(encoding0[1:][i])
                    index.append(i)

            # fc8与fc7[0]做欧式距离，fc8为0的一行跳过
            fc8 = np.asarray(fc8)
            if len(fc8) == 0: exit(-1)

            error = fc8 - encoding0[0]

            Eu_dis = np.sum(error ** 2, 1) / (m * k)  # [N-1,]

            Eu_dis_2=np.ones([N-1,])*10 # 也可以按照该字段 从小到大排序
            for i in range(len(Eu_dis)):
                Eu_dis_2[index[i]]=Eu_dis[i]


            Eu_dis = np.asarray((Eu_dis < 0.1), np.uint8)  # [x,] 0.065～0.1
            print('Eu_dis', Eu_dis)
            # 结合欧式距离，返回到源图的序号上
            encoding_3 = encoding_2.copy()
            # --------------------------------------------------------

            for i in range(len(Eu_dis)):
                if not Eu_dis[i]:
                    encoding_3[index[i]] = 0

            print('encoding_3', encoding_3)

            labels_1 = np.asarray((labels[1:N] - labels[0]) == 0, np.uint8)  # [N-1,]

            encoding_4 = encoding_3 * labels_1  # 编码一样，还有保证对应的真实标签是一样的（否则说明编码不对）

            comp1_cls_test = np.zeros((N - 1, 7))
            comp1_cls_test[:, 0] = score_1
            comp1_cls_test[:, 1] = encoding_4
            comp1_cls_test[:, 2] = encoding_2_1
            comp1_cls_test[:, 3] = labels_1
            comp1_cls_test[:, 4] = encoding_3
            comp1_cls_test[:,5]=Eu_dis_2
            comp1_cls_test[:,6]=encoding_8

            # 按score 字段排序
            import pandas as pd

            comp1_cls_test = pd.DataFrame(comp1_cls_test,
                                          columns=['score', 'encoding', 'True encoding', 'True label', 'Eu_dis','Eu_error','same_clos_num'])

            # comp1_cls_test = comp1_cls_test.sort_values('score', ascending=False)
            comp1_cls_test = comp1_cls_test.sort_values('Eu_error', ascending=True)
            print(comp1_cls_test[:50])
            # exit(-1)
            # 根据这个排序表，计算Precision，Recall ，计算top-10的结果
            Precision = np.sum(comp1_cls_test.values[:10][:, 1]) / 10
            print('Precision', Precision)

            Recall = np.sum(comp1_cls_test.values[:10][:, 1]) / np.sum(comp1_cls_test.values[:, 1])
            print('Recall', Recall)

            # 计算ap
            ap = voc_ap([Recall], [Precision])
            print('ap', ap)

        # 预测
        if train == -1:
            # 每个类别计算ap，所有类别取平均得到mAp
            def voc_ap(rec, prec, use_07_metric=False):
                """Compute VOC AP given precision and recall. If use_07_metric is true, uses
                the VOC 07 11-point method (default:False).
                """
                if use_07_metric:
                    # 11 point metric
                    ap = 0.
                    for t in np.arange(0., 1.1, 0.1):
                        if np.sum(rec >= t) == 0:
                            p = 0
                        else:
                            p = np.max(prec[rec >= t])
                        ap = ap + p / 11.
                else:
                    # correct AP calculation
                    # first append sentinel values at the end
                    mrec = np.concatenate(([0.], rec, [1.]))
                    mpre = np.concatenate(([0.], prec, [0.]))

                    # compute the precision envelope
                    for i in range(mpre.size - 1, 0, -1):
                        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

                    # to calculate area under PR curve, look for points
                    # where X axis (recall) changes value
                    i = np.where(mrec[1:] != mrec[:-1])[0]

                    # and sum (\Delta recall) * prec
                    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
                return ap


            data = Data('test.data', batch_size)  # train.data
            data.shuttle_data()

            N = data.num_images
            feat = coding
            steps = N // batch_size
            # N=steps*batch_size
            # 样本较多，按批进行
            encoding = np.zeros((N, m * k))  # 记录每张图片的编码  [64,2048]
            labels = np.zeros((N,), np.int64)  # 记录每张图的真实标签 [64,]
            # score = np.zeros((N, num_classes))  # 记录每张图片的分成每一类的概率值  [64,10]
            # cols=c7.shape[-1]
            cols = fc7.get_shape().as_list()[-1]
            cols = int(cols)
            fc7_Eu_ = np.zeros((N, cols))  # [64,2048]

            for step in range(steps):
                batch_x, batch_y = data.Next_batch()

                feat_1 = sess.run(feat, {x: batch_x, dropout: 1.})  # 编码
                pred_1 = sess.run(pred, {x: batch_x, dropout: 1.})  # 预测类别
                fc7_1 = sess.run(fc7, {x: batch_x, dropout: 1.})
                encoding[step * batch_size:(step + 1) * batch_size] = feat_1
                # score[step * batch_size:(step + 1) * batch_size] = pred_1
                labels[step * batch_size:(step + 1) * batch_size] = batch_y
                fc7_Eu_[step * batch_size:(step + 1) * batch_size] = fc7_1

            # 假设第一张图为查询图片，记录其他图片分成该类别的概率，与该图编码一致记为1,否则为0,与该图的真实类相同记为1,否则为0;

            # score_1 = score[1:N, int(labels[0])]  # [N-1,] [63,]
            score_1 = np.zeros([N - 1, ])

            encoding0 = encoding.copy()

            # 根据编码粗搜索
            encoding = np.round(encoding).astype(np.uint8)  # [N,m*k]

            encoding_1 = np.bitwise_xor(encoding[1:], encoding[0])

            encoding_2 = np.asarray(np.sum(encoding_1, 1) < int(m * k * 2 * 0.18), np.uint8)  # [n-1,] 0.18～0.2

            encoding_8=np.sum(encoding_1, 1)

            encoding_2_1 = encoding_2.copy()

            # 精搜索
            '''
            # 取编码层的上一层做欧式距离

            fc8 = []
            index = []
            for i, x in enumerate(encoding_2):
                if x:
                    fc8.append(fc7_Eu_[1:][i])
                    index.append(i)

            # fc8与fc7[0]做欧式距离，fc8为0的一行跳过
            fc8=np.asarray(fc8)
            if len(fc8)==0:exit(-1)

            error=fc8-fc7_Eu_[0]

            error=error**2

            Eu_dis=np.sum(error,1)/cols
            print('Eu_dis', Eu_dis.shape)
            # Eu_dis=np.sum((np.squeeze(fc8,1)-fc7[0])**2,1)#/cols

            Eu_dis=np.asarray((Eu_dis < 0.16), np.uint8) # [x,] 0.1～0.25
            print('Eu_dis', Eu_dis)
            # 结合欧式距离，返回到源图的序号上
            encoding_3=encoding_2.copy()
            '''
            # ---------------------------------------------------------

            # 使用编码层求欧式距离
            fc8 = []
            index = []
            for i, x in enumerate(encoding_2):
                if x:
                    fc8.append(encoding0[1:][i])
                    index.append(i)

            # fc8与fc7[0]做欧式距离，fc8为0的一行跳过
            fc8 = np.asarray(fc8)
            if len(fc8) == 0: exit(-1)

            error = fc8 - encoding0[0]

            Eu_dis = np.sum(error ** 2, 1) / (m * k)  # [N-1,]

            Eu_dis_2=np.ones([N-1,])*10 # 也可以按照该字段 从小到大排序
            for i in range(len(Eu_dis)):
                Eu_dis_2[index[i]]=Eu_dis[i]


            Eu_dis = np.asarray((Eu_dis < 0.1), np.uint8)  # [x,] 0.065～0.1
            print('Eu_dis', Eu_dis)
            # 结合欧式距离，返回到源图的序号上
            encoding_3 = encoding_2.copy()
            # --------------------------------------------------------

            for i in range(len(Eu_dis)):
                if not Eu_dis[i]:
                    encoding_3[index[i]] = 0

            print('encoding_3', encoding_3)

            labels_1 = np.asarray((labels[1:N] - labels[0]) == 0, np.uint8)  # [N-1,]

            encoding_4 = encoding_3 * labels_1  # 编码一样，还有保证对应的真实标签是一样的（否则说明编码不对）

            comp1_cls_test = np.zeros((N - 1, 7))
            comp1_cls_test[:, 0] = score_1
            comp1_cls_test[:, 1] = encoding_4
            comp1_cls_test[:, 2] = encoding_2_1
            comp1_cls_test[:, 3] = labels_1
            comp1_cls_test[:, 4] = encoding_3
            comp1_cls_test[:,5]=Eu_dis_2
            comp1_cls_test[:,6]=encoding_8

            # 按score 字段排序
            import pandas as pd

            comp1_cls_test = pd.DataFrame(comp1_cls_test,
                                          columns=['score', 'encoding', 'True encoding', 'True label', 'Eu_dis','Eu_error','same_clos_num'])

            # comp1_cls_test = comp1_cls_test.sort_values('score', ascending=False)
            comp1_cls_test = comp1_cls_test.sort_values('Eu_error', ascending=True)
            print(comp1_cls_test[:50])
            # exit(-1)
            # 根据这个排序表，计算Precision，Recall ，计算top-10的结果
            Precision = np.sum(comp1_cls_test.values[:10][:, 1]) / 10
            print('Precision', Precision)

            Recall = np.sum(comp1_cls_test.values[:10][:, 1]) / np.sum(comp1_cls_test.values[:, 1])
            print('Recall', Recall)

            # 计算ap
            ap = voc_ap([Recall], [Precision])
            print('ap', ap)
