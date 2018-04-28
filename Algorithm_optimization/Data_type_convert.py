# -*- coding:utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import os
import csv

data=pickle.load(open("./explor_17_crop_all_features.pkl","rb"),encoding="iso8859-1") # pickle3 打开pickle2

img_path=list(data.keys()) # 567297
img_feature=np.asarray(list(data.values()),np.float32) # [567297,128]
img_feature=np.squeeze(img_feature)

# img_path=data.keys()
# img_feature=data.values()
# print(len(img_path),len(img_feature)) # 567297,567297
# print(type(img_path),type(img_feature)) # <class 'dict_keys'>,<class 'dict_values'>
num_images=len(img_path)

# query_picture=np.random.choice(num_images,10) # 随机查找10张查询影像
# print(query_picture)
query_picture=[480078,539237,120465,352352,489944,401508,501127,330056,240369,44489]

# 获取查询影像的图片名字
query_name=[]

# 计算查询影像与源图片的欧式距离
comp1_cls=np.zeros((num_images, len(query_picture)))
for i in range(len(query_picture)):
    Euclidean_distance=np.sum((img_feature-img_feature[query_picture[i]])**2,1) # [567297,]
    comp1_cls[:, i] = Euclidean_distance
    query_name.append(img_path[query_picture[i]])

# print(query_name)
comp1_cls = pd.DataFrame(comp1_cls,columns=range(len(query_picture)))

# pandas 加上一列，图片路径
comp1_cls['img_path']=img_path

# 将结果保持到文件中
# 保留查询影像文件名
with open(os.path.join('./output','query_image.data'),'w') as fp:
    # writer = csv.writer(fp)
    # writer.writerows(query_name)
    for i in range(len(query_name)):
        fp.write(query_name[i]+'\n')
    # fp.writelines(query_name)

comp1_cls = comp1_cls.sort_values(0, ascending=True) # 从小到大排序

# for j in range(len(query_picture)):
with open('feature'+str(0)+'.data','w') as fp:
        pd.DataFrame(comp1_cls).to_csv(fp)#,index=False,header=False)
