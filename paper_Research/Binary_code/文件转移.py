# -*- coding:utf-8 -*-

import os
import glob
import csv
import random
import shutil

# 文件转移
path = './train/*'
path_t='./train_1'
if not os.path.exists(path_t):os.mkdir(path_t)
image_paths = glob.glob(path)

i=0
for path in image_paths:
    path2=glob.glob(os.path.join(path,'*'))
    if len(path2)>50:
        if not os.path.exists(os.path.join(path_t,str(i))): os.mkdir(os.path.join(path_t,str(i)))
        for path3 in path2:
            shutil.copy(path3,os.path.join(path_t,str(i)))

        i+=1
