# -*- coding:utf-8 -*-

import os
import glob
import csv
import random

# 数据接口 得到的数据格式 xxxx/xxx/xx.jpg id (后面为该图像id)
# 写入到一个文件中
path = './train/*/*'

image_paths = glob.glob(path)

datas = []
labels=[]
for path in image_paths:
    label = int(path.split('/')[-2])
    if label not in labels:
        labels.append(label)
labels=sorted(labels) # 排序
label_dict=dict(zip(labels,range(len(labels))))


for path in image_paths:
    label = int(path.split('/')[-2])
    datas.append([path, label_dict[label]])
m=len(datas) # 图片数

# 0.8做train，0.2做valid
random.shuffle(datas) # 随机打乱数据

with open('train.data', 'w') as fp:
    writer=csv.writer(fp)
    writer.writerows(datas[:int(0.8*m)])

with open('valid.data', 'w') as fp:
    writer=csv.writer(fp)
    writer.writerows(datas[int(0.8*m):])

# -----------------------------------
path = './test/*/*'
image_paths = glob.glob(path)
datas = []
for path in image_paths:
    label = int(path.split('/')[-2])
    datas.append([path, label])
print(len(datas))
with open('test.data', 'w') as fp:
    writer=csv.writer(fp)
    writer.writerows(datas[int(0.8*m):])
