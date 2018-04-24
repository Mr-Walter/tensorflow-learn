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
        image = tf.random_crop(image, size=[128, 64, num_channels])

        # Randomly flip the image horizontally.
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        # Randomly adjust hue, contrast and saturation.
        image = tf.image.random_hue(image, max_delta=0.05)
        image = tf.image.random_contrast(image, lower=0.3, upper=1.0)
        image = tf.image.random_brightness(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.0, upper=2.0)
        image = tf.image.rot90(image, np.random.randint(1,4))
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

sess=tf.InteractiveSession()
for path in flod_paths:
    i=0
    # img_paths=glob.glob(os.path.join(path,'*'))
    img_paths=os.listdir(path)
    for _ in range(1000//len(img_paths)):
        j=0
        for img_path in img_paths:
            # 进度条
            sys.stdout.write('>>>正在处理中: %.2f%%\r'%((i+1)/len(img_paths)))
            sys.stdout.flush()
            img=pre_process_image(skimage.io.imread(os.path.join(path,img_path)),training)
            # img=preprocess_for_train(skimage.io.imread(os.path.join(path,img_path)))
            skimage.io.imsave(os.path.join(path,str(j)+'_'+str(i)+'.jpg',),sess.run(img))
            i+=1
        i=0
        j+=1
sess.close()

sys.stdout.close()
sys.exit(0)
