# -*- coding:utf-8 -*-

import pickle
import numpy as np
import pandas as pd
import time

# start_time=datetime.datetime.now()
start_time=time.clock()

with open("./int8_explor_17_crop_all_features.pkl","rb") as fp:
    data=pickle.load(fp,encoding="iso8859-1") # pickle3 打开pickle2

img_path=list(data.keys()) # 567297
img_feature=np.asarray(list(data.values()),np.uint8) # [567297,128]
img_feature=np.squeeze(img_feature)

num_images=len(img_path)

query_picture=np.random.choice(num_images,100) # 随机查找20张查询影像

# img_feature=(2.*img_feature/256-1).astype(np.float32)
img_feature=(1.*img_feature/128).astype(np.float32)

''' 执行100次  [56w,128]x[128,1]
comp1_cls = np.zeros((num_images, 2))
# 计算查询影像与源图片的欧式距离
for i in range(len(query_picture)):
    # Euclidean_distance=np.sum((img_feature-img_feature[query_picture[i]])**2,1) # [567297,]
    comp1_cls[:, 0] = np.dot(img_feature,img_feature[query_picture[i]])*(-1)
    comp1_cls[:, 1] =np.dot(img_feature,img_feature[query_picture[i]])*(-1)

    # comp1_cls = pd.DataFrame(comp1_cls,columns=['float','int'])
    #
    # comp1_cls = comp1_cls.sort_values('float', ascending=True) # 从小到大排序
    # index_float=comp1_cls.index[:2000] # 获取2000个行索引号
    #
    # comp1_cls = comp1_cls.sort_values('int', ascending=True) # 从小到大排序
    # index_int=comp1_cls.index[:2000]
    # # 比较2个索引号，有多少个是一样
    # # 首先把两个list转换成set，然后对两个set取交集，即可得到两个list的重复元素
    # set_1=set(list(index_float))
    # set_2=set(list(index_int))
    #
    # s=set_1 & set_2

    #print(len(s),end=',')
'''

# 执行一次 [56w,128,128,100]
query_matrix=img_feature[query_picture] # [100,128]
comp1_cls=np.dot(img_feature,query_matrix.T) # [56w,100]

end_time=time.clock()

print('总时间：',end_time-start_time)
