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
batch_size = 64 # 逐步改变 128~128*4
img_Pixels_h = 128
img_Pixels_w = 64
img_Pixels_c = 3
num_classes = 10
epochs = arg.epochs
log_dir = './model'
keep_rata=0.4

m=10
k=256*2
max_entropy = m*np.log2(k)

def entropy(ivf_vecs): # ivf_vecs shape [batch_size,k]
  zero = tf.constant(1e-30, dtype=tf.float32)
  ivf_vecs+=zero;
  entropy = -tf.reduce_sum(tf.multiply(ivf_vecs, tf.log(ivf_vecs))) / (tf.cast(tf.shape(ivf_vecs)[0], tf.float32) * tf.log(tf.cast(2,tf.float32)))
  return entropy

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

def distort_location(image):
    image=tf.image.rot90(image, 1)
    image=tf.image.flip_left_right(image)
    image = tf.image.flip_up_down(image)
    # net = tf.image.per_image_standardization(net)
    return image #tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image, height, width, bbox):
    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    #if image.dytpe != tf.float32:
    #    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)
    #distorted_image = tf.image.resize_images(distorted_image, height, width, method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    return distorted_image

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

        x1=[]
        # x2=[]
        for i in range(net.shape[0]):
            # boxes = tf.constant([[[0.05, 0.05, 0.9, 0.7], [0.35, 0.47, 0.5, 0.56]]])
            x2=tf.squeeze(tf.slice(net, [i, 0, 0, 0], [1, img_Pixels_h, img_Pixels_w, img_Pixels_c]), 0)
            x1.append(preprocess_for_train(x2, img_Pixels_h, img_Pixels_w, None))

            # 数据增强
        net=tf.convert_to_tensor(x1,tf.float32)
        # x2 = tf.convert_to_tensor(x2, tf.float32)
        # net=tf.concat([net,x1,x2],0) # [32*5,128,64,3]

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

        # net = slim.fully_connected(net, 512, activation_fn=tf.nn.relu,trainable=self.trainable, scope='fc1')

        net= slim.fully_connected(net,m*k,activation_fn=tf.nn.relu,scope='coding_layer')

        feat = tf.split(net, num_or_size_splits=m, axis=1)
        out = [None] * m
        for i in range(m):
            out[i] = tf.nn.softmax(feat[i])
        self.net = tf.concat(out, axis=1) # [n,m*k]

        self.pred=slim.fully_connected(net,num_classes,activation_fn=tf.nn.softmax,scope='output')


if __name__=="__main__":
    data=Data('train.data',batch_size)
    # data.Next_batch()

    x = tf.placeholder(tf.float32, (batch_size, img_Pixels_h, img_Pixels_w, img_Pixels_c))
    y = tf.placeholder(tf.int64, (None,))
    dropout=tf.placeholder(tf.float32)

    net=Net(x,dropout,flag)
    coding=net.net
    pred=net.pred
    '''
    # 获取变量
    graph = tf.get_default_graph()
    # 通过变量名获取变量
    w = graph.get_tensor_by_name("output/weights:0")  # [2048,12]
    b = graph.get_tensor_by_name("output/biases:0")  # [12,]

    # y [n,] -->[n,12]
    y3 = tf.one_hot(y, num_classes)
    x_ = tf.matmul(tf.matmul((y3 - b), tf.transpose(w)), tf.matrix_inverse(tf.matmul(w, tf.transpose(w))))

    cost0 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=x_, logits=coding))
    '''

    # -----------------------------------
    # Entropy losses
    sb_codes = tf.split(coding, num_or_size_splits=m, axis=1)  # m个 [batch_size,k] list
    sb_entropy = [None] * m
    sb_b = [None] * m
    sb_batch_ent = [None] * m
    batch_entropy = 0
    ind_entropy = 0

    for i in range(m):
        sb_entropy[i] = entropy(sb_codes[i])
        sb_b[i] = (tf.reduce_sum(sb_codes[i], 0) / tf.cast(tf.shape(sb_codes[i])[0], tf.float32)) + tf.constant(
            1e-30, dtype=tf.float32)
        sb_batch_ent[i] = -tf.reduce_sum(tf.multiply(sb_b[i], tf.log(sb_b[i]))) / tf.log(tf.cast(2, tf.float32))
        batch_entropy += sb_batch_ent[i]
        ind_entropy += sb_entropy[i]

    batch_entropy = batch_entropy / max_entropy
    ind_entropy = ind_entropy / max_entropy

    # Loss
    # loss_cl = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_pl)) / np.log(
    #     nclass)
    # acc1 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_pl, 1), tf.float32))
    # acc5 = tf.reduce_mean(tf.cast(tf.nn.in_top_k(logits, labels_pl, 5), tf.float32))
    # loss = loss_cl + 1. * ind_entropy - 1. * batch_entropy  # 总loss
    '''
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
    '''
    # -------------------------
    # y_ = np.concatenate([y, y, y], 0)  # [32*5,128,64,3]
    '''
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

    # cost2_ = tf.losses.mean_squared_error(net1, net2)/(m*k)
    net1 = tf.split(net1, num_or_size_splits=m, axis=1)
    net2 = tf.split(net2, num_or_size_splits=m, axis=1)
    cost2 = 0
    for i in range(m):
        cost2 += tf.losses.mean_squared_error(net1[i], net2[i])
    cost2 = cost2 / m
    
    # label不同时,cost最大，倒数最小(转成求倒数，因为最后要求的是整体cost 最小)
    error = tf.cast(tf.logical_not(tf.equal(y1, y2)), tf.float32)  # [n-1,]
    net1 = tf.slice(net, [0, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]
    net2 = tf.slice(net, [1, 0], [batch_size - 1, net.shape[-1]])  # [n-1,32]

    error = tf.expand_dims(error, -1)  # [n-1,1]
    net1 = net1 * error
    net2 = net2 * error

    # cost3_ = tf.losses.mean_squared_error(net1, net2)/(m*k)
    net1 = tf.split(net1, num_or_size_splits=m, axis=1)
    net2 = tf.split(net2, num_or_size_splits=m, axis=1)
    cost3 = 0
    for i in range(m):
        cost3 += tf.losses.mean_squared_error(net1[i], net2[i])
    cost3 = cost3 / m
    # '''

    '''
    # 同一编码内只能出现一个大于0.5的数
    net = tf.split(coding, num_or_size_splits=m, axis=1)
    cost4 = 0
    # cost5=0
    for i in range(m):
        cost4 += tf.reduce_mean(tf.square(tf.reduce_sum(tf.round(net[i]), 1) - 1))
    cost4 = cost4*10000 # 加上个大点的权重，这样防止coding层全为小于0.5
    # '''
    cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))/np.log(
        num_classes)

    # cost = cost + cost2 + 1. / cost3  # cost最小 ，则cost3 最大
    cost = cost + 1. * ind_entropy - 1. * batch_entropy #+cost2 +1./cost3 #+cost4   # 总loss

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
                    # batch_y = list(batch_y) * 3
                    # batch_y = np.asarray(batch_y, np.int64)

                    _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y,dropout:keep_rata})
                    epoch_loss += c

                    if step % 50 == 0:
                        print('epoch', epoch, '|', 'step:', step, '|', 'Train acc: ',
                              accuracy.eval({x: batch_x, y: batch_y,dropout:keep_rata}), '|', 'loss:', c)

                print(epoch, ' : ', epoch_loss / steps)
                save_path = saver.save(sess, os.path.join(log_dir, 'model.ckpt'))  # , global_step=epoch)
                print('model save to ', save_path)

        '''
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
        '''

        if train==0:
            data = Data('test.data', batch_size)
            # batch_x, batch_y = data.Next_batch()
            feat=coding
            N=data.num_images # 样本数
            dataX = np.zeros((N, m*k))
            label_dim=1
            if (label_dim == 1):
                labels = np.zeros((N,))
            else:
                labels = np.zeros((N, label_dim))

            nbatch = int(N / batch_size)
            bsize=batch_size
            for i in range(nbatch):
                images, labels[i * bsize:(i + 1) * bsize] = data.Next_batch()
                feed = {x: images}
                dataX[i * bsize:(i + 1) * bsize] = sess.run(feat, feed_dict=feed)
            testx, testy=dataX[:N].astype('float32'), labels[:N]

            split = 1
            mlabel = False;
            ldim = 1


            def split_dataset(testx, testy, split, nclass=10):
                ty = testy.astype(int)
                frac = split / nclass
                split = frac * nclass
                Dx = [];
                Dy = []
                Qx = [];
                Qy = []
                count = np.zeros((nclass,), dtype=int)
                for i in range(ty.shape[0]):
                    if (count[ty[i]] < frac):
                        Qx.append(testx[i])
                        Qy.append(testy[i])
                        count[ty[i]] += 1
                    else:
                        Dx.append(testx[i])
                        Dy.append(testy[i])
                return np.concatenate((Qx, Dx), axis=0), np.concatenate((Qy, Dy), axis=0), split

            if (not mlabel):  # reordering testx and testy as [query_set, database] with equal number of queries per class.
                testx, testy, split = split_dataset(testx, testy, split, 100)
            x = testx[split:]
            q = testx[:split]

            def retrieve(m, k, x, q, labelq, labelx, multilabel=False):
                code = np.zeros((m, x.shape[0]), dtype=int)
                approx_dis = np.zeros((q.shape[0], code.shape[1]))
                for i in range(m):
                    code[i, :] = np.argmax(x[:, i * k:(i + 1) * k], axis=1)
                    for j in range(q.shape[0]):
                        approx_dis[j] += q[j, i * k + code[i]]
                qres = np.argsort(-approx_dis, axis=1)
                print("mAP=%f" % get_results(qres, labelq, labelx, multilabel))
                return qres, approx_dis


            def get_results(res, labelq, labelx, multilabel=True, n=21):
                if (multilabel):
                    pr, rec = prec_recall_multiclass(res, labelq, labelx)
                else:
                    pr, rec = prec_recall(res, labelq, labelx)
                ap = 0
                for i in range(labelq.shape[0]):
                    ap += compute_ap(pr[i], rec[i], n)
                return ap / labelq.shape[0]


            def prec_recall(res, labelq, labelx):
                prec = np.zeros_like(res, dtype=float)
                recall = np.zeros_like(res, dtype=float)
                n = np.arange(res.shape[1]) + 1.0
                for i in range(res.shape[0]):
                    x = np.cumsum((labelx[res[i]] == labelq[i]).astype(float))
                    prec[i] = x / n
                    recall[i] = x / x[-1]
                return prec, recall


            def compute_ap(prec, rec, n):
                ap = 0;
                for t in np.linspace(0, 1, n):
                    all_p = prec[rec >= t];
                    if all_p.size == 0:
                        p = 0;
                    else:
                        p = all_p.max()
                    ap = 1.0 * ap + 1.0 * p / n;
                return ap

            def prec_recall_multiclass(res, labelq, labelx):
                prec = np.zeros_like(res, dtype=float)
                recall = np.zeros_like(res, dtype=float)
                n = np.arange(res.shape[1]) + 1.0
                for i in range(res.shape[0]):
                    x = np.cumsum((labelx[res[i]].dot(labelq[i]) > 0).astype(float))
                    prec[i] = x / n
                    recall[i] = x / x[-1]
                return prec, recall

            res, dis = retrieve(m, k, x, q, testy[:split], testy[split:], multilabel=mlabel)

            # print(res,dis)

        if train==-1:
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
            # 参考：http://blog.sina.com.cn/s/blog_9db078090102whzw.html
            # 计算所有图片分成某一类别的score（即softmax那层，每个维度的概率）  格式： id,score,ground truth label

            data = Data('valid.data', batch_size) # train.data
            data.shuttle_data()
            # batch_size=141
            # batch_x, batch_y = data.Next_batch
            N=data.num_images
            feat = coding
            steps = N // batch_size
            # 样本较多，按批进行
            encoding = np.zeros((N, m*k)) # 记录每张图片的编码
            labels = np.zeros((N,),np.int64) # 记录每张图的真实标签
            score = np.zeros((N,num_classes)) # 记录每张图片的分成每一类的概率值

            for step in range(steps):
                batch_x, batch_y = data.Next_batch()

                # feat = tf.round(feat) # [n,m*k]
                # feat=tf.argmax(tf.reshape(feat,[-1,m,k]),-1)
                feat_1 = sess.run(feat, {x: batch_x, dropout: 1.}) # 编码
                pred_1=sess.run(pred,{x: batch_x, dropout: 1.}) # 预测类别

                encoding[step*batch_size:(step+1)*batch_size]=feat_1
                score[step*batch_size:(step+1)*batch_size]=pred_1
                labels[step * batch_size:(step + 1) * batch_size]=batch_y

            # 假设第一张图为查询图片，记录其他图片分成该类别的概率，与该图编码一致记为1,否则为0,与该图的真实类相同记为1,否则为0;
            score_1=score[1:N,int(labels[0])] # [N-1,]

            '''
            # print(np.sum(encoding[:20],1))
            encoding=np.argmax(np.reshape(encoding,[-1,m,k]),-1) # [N,m]
            encoding_1=np.asarray((encoding[1:N]-encoding[0]) == 0, np.uint8) # [N-1,m]
            print(encoding[:50])       
            # 按列累加，取最后一列，如果值为m，说明前面的都为1,即编码正确
            encoding_2=np.asarray(np.cumsum(encoding_1, -1)[:,-1]>int(m*0.5),np.uint8)
            encoding_2_1 = encoding_2
            # '''
            '''
            # ------------------
            encoding0 = np.reshape(encoding, [-1, m, k]) # [N,m,k]
            encoding = np.argmax(encoding0, -1)  # [N,m]
            # encoding_1 = np.asarray((encoding[1:N] - encoding[0]) == 0, np.uint8)  # [N-1,m]
            # print(encoding[:50])
            encoding0=encoding0[0] # [1,m]  查询图片
            # print(encoding0)
            '''
            '''
            score_4=[]
            for j in range(1,N):
                score_3 = 0
                for i in range(m):
                    score_3+=encoding0[i,encoding[j,i]]
                score_4.append(score_3) # N-1

            encoding_2 = np.asarray(score_4, np.float32)
            encoding_2 = np.asarray((encoding_2 > 0.1), np.uint16)
            encoding_2_1 = encoding_2
            # '''
            '''
            score_4 = []
            for j in range(1, N):
                score_3 = 0
                for i in range(m):
                    score_3 += encoding0[j,i,encoding[i]]
                score_4.append(score_3)  # N-1

            encoding_2 = np.asarray(score_4, np.float32)

            encoding_2=np.asarray((encoding_2>0.1),np.uint16)

            encoding_2_1 = encoding_2
            '''
            # ------------------
            '''
            encoding = np.round(encoding,0) # [N,m*k]
            encoding_1 = np.asarray((encoding[1:N] - encoding[0]) == 0, np.uint8)  # [N-1,m*k]

            # 按列累加，取最后一列，如果值为m，说明前面的都为1,即编码正确
            encoding_2 = np.asarray(np.cumsum(encoding_1, -1)[:, -1] == m*k, np.uint8)
            '''
            # '''
            # print(encoding[:20])
            encoding_1 = np.asarray((encoding[1:N] - encoding[0]))
            encoding_2=np.sum(encoding_1 ** 2, 1) / (m*k) # [N-1,]

            encoding_2 = np.asarray((encoding_2 < 0.001), np.uint8)  # [N-1,m*k]
            encoding_2_1=encoding_2
            # '''
            labels_1=np.asarray((labels[1:N]-labels[0])==0,np.uint8) # [N-1,]

            encoding_2=encoding_2*labels_1 # 编码一样，还有保证对应的真实标签是一样的（否则说明编码不对）

            comp1_cls_test=np.zeros((N-1,4))
            comp1_cls_test[:,0]=score_1
            comp1_cls_test[:,1]=encoding_2
            comp1_cls_test[:, 2] = encoding_2_1
            comp1_cls_test[:, 3] = labels_1
            # 按score 字段排序
            import pandas as pd
            comp1_cls_test = pd.DataFrame(comp1_cls_test, columns=['score', 'encoding','True encoding','True label'])

            comp1_cls_test = comp1_cls_test.sort_values('score', ascending=False)
            print(comp1_cls_test[:10])
            # exit(-1)
            # 根据这个排序表，计算Precision，Recall ，计算top-10的结果
            Precision=np.sum(comp1_cls_test.values[:10][:,1])/10
            print('Precision',Precision)

            Recall=np.sum(comp1_cls_test.values[:10][:,1])/np.sum(comp1_cls_test.values[:,1])
            print('Recall', Recall)

            # 计算ap
            ap=voc_ap([Recall],[Precision])
            print('ap',ap)
