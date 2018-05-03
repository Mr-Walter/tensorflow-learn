# -*- coding:utf-8 -*-

import pickle
import numpy as np

with open("./uint8_explor_17_crop_all_features.pkl","rb") as fp:
    data = pickle.load(fp,encoding="iso8859-1")

# with open("./explor_17_crop_all_features.pkl","rb") as fp:
#     data=pickle.load(fp,encoding="iso8859-1") # pickle3 打开pickle2

img_path=list(data.keys()) # 567297
img_feature=np.asarray(list(data.values()),np.float32) # [567297,128]
img_feature=np.squeeze(img_feature)
print(img_feature.shape)
