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

        
        
#################

# -*- coding:utf-8 -*-

import os
import glob
import csv
import random
import shutil

# 文件转移
path = './gt_bbox/*'
path_t='./train'
if not os.path.exists(path_t):os.mkdir(path_t)
image_paths = glob.glob(path)

for path in image_paths:
    img_name=path.split('/')[-1]
    if not img_name.endswith('.jpg'): continue
    img_id=int(img_name.split('_')[0])
    if not os.path.exists(os.path.join(path_t,str(img_id))):os.mkdir(os.path.join(path_t,str(img_id)))
    shutil.copy(path, os.path.join(path_t,str(img_id),img_name))
