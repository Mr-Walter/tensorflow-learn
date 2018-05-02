# -*- coding:utf-8 -*-

import pickle
import numpy as np
import pandas as pd
# import os
# import csv
# import datetime
import time
import cv2

data=pickle.load(open("./explor_17_crop_all_features.pkl","rb"),encoding="iso8859-1") # pickle3 打开pickle2

img_path=list(data.keys()) # 567297
img_feature=np.asarray(list(data.values()),np.float32) # [567297,128]
img_feature=np.squeeze(img_feature)

# img_path=data.keys()
# img_feature=data.values()
# print(len(img_path),len(img_feature)) # 567297,567297
# print(type(img_path),type(img_feature)) # <class 'dict_keys'>,<class 'dict_values'>
num_images=len(img_path)

# query_picture=np.random.choice(num_images,1) # 随机查找10张查询影像
# print(query_picture)
# query_picture=[480078,539237,120465,352352,489944,401508,501127,330056,240369,44489]
query_picture=[480078]

# 获取查询影像的图片名字
query_name=[]

# 转成int编码
img_feature2=(img_feature*128).astype(np.int8) # [-128,127]
# img_feature2=((img_feature+1)/2*128).astype(np.uint8) # [0,255]

# 计算欧式距离
def distance(x,y):
    # x=x.astype(np.int16)
    # y=y.astype(np.int16)
    l = list([y]) * len(x)
    l = np.asarray(l, np.uint8)
    c = cv2.absdiff(x, l)
    return np.sum(c**2,1,np.float32)
    # x=x.astype(np.int32)
    # return np.sum((x-y)**2, 1, np.float32)

# 计算查询影像与源图片的欧式距离
# start_time=datetime.datetime.now()
start_time=time.clock()

comp1_cls=np.zeros((num_images, len(query_picture)+1))
img_feature2=img_feature2.astype(np.int16)
for i in range(len(query_picture)+100):
    # Euclidean_distance=np.sum((img_feature-img_feature[query_picture[0]])**2,1) # [567297,]
    # Euclidean_distance=distance(img_feature,img_feature[query_picture[0]])
    # Euclidean_distance=np.sum(np.square(np.subtract(img_feature2, img_feature2[query_picture[0]], dtype=np.float32)), 1, dtype=np.float32)
    # comp1_cls[:, i] = Euclidean_distance
    #
    # comp1_cls[:, i+1]=distance(img_feature2,img_feature2[query_picture[i]])
    # Euclidean_distance=distance(img_feature2, img_feature2[query_picture[0]])
    # Euclidean_distance = np.sum((img_feature2 - img_feature2[query_picture[0]]) ** 2, 1,np.float16)  # [567297,]
    Euclidean_distance= np.mean((img_feature2 - img_feature2[query_picture[0]]) ** 2, 1, np.float16)
    # Euclidean_distance = np.sum(abs(img_feature2-img_feature2[query_picture[0]]), 1, np.float32)

    # err = img_feature2 - img_feature2[query_picture[0]]
    # mean_err = np.mean(err, 1)
    # Euclidean_distance = np.mean((err - mean_err[:, np.newaxis]) ** 2, 1, np.float32)


    # Euclidean_distance=np.sum(np.square(np.subtract(img_feature2, img_feature2[query_picture[0]], dtype=np.int32)), 1, dtype=np.float32)

    # query_name.append(img_path[query_picture[i]])

# end_time=datetime.datetime.now()
# print((end_time-start_time).seconds)
end_time=time.clock()
print(end_time-start_time)
exit(0)

# print(query_name)
comp1_cls = pd.DataFrame(comp1_cls,columns=['float','int'])

# pandas 加上一列，图片路径
comp1_cls['img_path']=img_path

comp1_cls = comp1_cls.sort_values('float', ascending=True) # 从小到大排序
index_float=comp1_cls.index[:2000] # 获取2000个行索引号

# for j in range(len(query_picture)):
# with open('./output/float'+str(0)+'.data','w') as fp:
#         pd.DataFrame(comp1_cls).to_csv(fp)#,index=False,header=False)

comp1_cls = comp1_cls.sort_values('int', ascending=True) # 从小到大排序
index_int=comp1_cls.index[:2000]
# with open('./output/int' + str(0) + '.data', 'w') as fp:
#     pd.DataFrame(comp1_cls).to_csv(fp)  # ,index=False,header=False)

# 比较2个索引号，有多少个是一样
# 首先把两个list转换成set，然后对两个set取交集，即可得到两个list的重复元素
set_1=set(list(index_float))
set_2=set(list(index_int))

s=set_1 & set_2

print(len(s))
