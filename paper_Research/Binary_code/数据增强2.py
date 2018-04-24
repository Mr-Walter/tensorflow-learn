# -*- coding：utf-8 -*-

import tensorflow as tf
import skimage.io
import glob
import os
import numpy as np
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # 指定GPU from 0,1,2....

img_size_cropped=128
num_channels=3
training=True
def pre_process_image(image, training):

    if training:
        # For training, add the following to the TensorFlow graph.
        # Randomly crop the input image.
        image = tf.random_crop(image, size=[128, 64,3])
        # image = tf.image.resize_image_with_crop_or_pad(image,
        #                                                target_height=128,
        #                                                target_width=128)

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        # image = tf.image.rot90(image, np.random.randint(1,4))
        # image=tf.clip_by_value(image,0.,255.)
    else:
        image = tf.image.resize_image_with_crop_or_pad(image,
                                                       target_height=128,
                                                       target_width=64)

    return image

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
    return image # tf.clip_by_value(image, 0.0, 1.0)

def preprocess_for_train(image):#, height, width, bbox):
    # if bbox is None:
    #     bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    # if image.dytpe != tf.float32:
    #    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes=bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)
    distorted_image = tf.random_crop(image, size=[128, 64, num_channels])
    # distorted_image = tf.image.resize_images(image, (height, width), method=np.random.randint(4))
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    distorted_image = tf.image.random_flip_up_down(distorted_image)
    distorted_image = distort_color(distorted_image, np.random.randint(4))
    distorted_image-tf.image.rot90(distorted_image,np.random.randint(4))
    return distorted_image


flod_paths=glob.glob('./train_1/*')

# Launch the graph
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7  # 占用GPU70%的显存

sess=tf.InteractiveSession(config=config)
for path in flod_paths:
    # i=0
    # img_paths=glob.glob(os.path.join(path,'*'))

    # while True:
    img_paths = os.listdir(path)
    img_len = len(img_paths)
    imgs=[]
    i=0
    add_len=np.round(500/img_len).astype(np.int16) # 数据增强到500张
    for img_path in img_paths*add_len:
        # 进度条
        sys.stdout.write('>>>正在处理中: %.2f%%\r'%((i+1)/(img_len*add_len))*100)
        sys.stdout.flush()

        # tf内置方法读取jpg 参考：https://blog.csdn.net/wayne2019/article/details/77884478
        image_raw = tf.gfile.FastGFile(os.path.join(path,img_path), 'rb').read()  # bytes
        img = tf.image.decode_jpeg(image_raw)  # Tensor
        img = pre_process_image(img, training)
        imgs.append(img)
        i += 1
        # img=pre_process_image(skimage.io.imread(os.path.join(path,img_path)),training)
        # img=preprocess_for_train(skimage.io.imread(os.path.join(path,img_path)))
    imgs=tf.convert_to_tensor(imgs,dtype=tf.uint8).eval()
    [skimage.io.imsave(os.path.join(path,'_'+str(k)+'.jpg',),imgs[k]) for k in range(len(imgs))]


sess.close()

sys.stdout.close()
sys.exit(0)
