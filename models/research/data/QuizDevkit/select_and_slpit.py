
# coding: utf-8

import cv2
import os,shutil
from lxml import etree
import numpy as np
from matplotlib import pyplot as plt
import random

############################################## 第一步 剔除模糊图片 ########################################
# 图片筛选 ,剔除模糊的，考虑到拍摄的场景基本上都差不多，图片的大小也一样，这里才有一个7x7的高通滤波，求图片的锐度值。
# 按锐度值大小排序，删除锐度值较小的图片
cwd = os.getcwd()
xml_dir = os.path.join(cwd,'annotations')
img_dir = os.path.join(cwd,'images')
split_dir = os.path.join(cwd,'split')
rb_dir_img = os.path.join(cwd,'rubbish','image')
rb_dir_xml = os.path.join(cwd,'rubbish','xml')
if not os.path.exists(rb_dir):
    os.mkdir(rb_dir)


flt = [-0.001953, -0.015625, -0.044922, -0.060547, -0.044922, -0.015625, -0.001953,
        -0.015625,  -0.080078, -0.128906, -0.128906, -0.128906, -0.080078, -0.015625,
        -0.044922,  -0.128906,  0.091797,  0.349609, 0.091797,  -0.128906, -0.044922,
        -0.060547,  -0.128906,  0.349609,  0.835938, 0.349609,  -0.128906, -0.060547,
        -0.044922,  -0.128906,  0.091797,  0.349609, 0.091797,  -0.128906, -0.044922,
        -0.015625,  -0.080078, -0.128906, -0.128906, -0.128906, -0.080078, -0.015625,
        -0.001953, -0.015625, -0.044922, -0.060547, -0.044922, -0.015625, -0.001953,
        ]
flt = np.reshape(flt,(7,7))
print(flt)

lst = []
dict_lst = {}
img_list = os.listdir(img_dir)

for f in img_list:
    if '.jpg' in f:
        
        img = cv2.imread(os.path.join(img_dir,f),cv2.IMREAD_COLOR)

        edges = np.abs(cv2.filter2D(img,-1,flt))
        shapness = np.sum(edges)
        lst.append(shapness)
        dict_lst[shapness] = f
        print('process:',f,'shapness:',shapness)

        """
        plt.subplot(121),plt.imshow(img,cmap = 'gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(edges,cmap = 'gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
        break
        cv2.imshow('image',img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        break
        """
sort_idx = np.argsort(lst)

# 删除锐度值最小的10张图片，验证下来这种方法不可靠，最终选择删除10张图片
for i in range(10):
    shutil.move(os.path.join(img_dir,dict_lst[lst[sort_idx[i]]]),rb_dir_img)
    shutil.move(os.path.join(xml_dir,dict_lst[lst[sort_idx[i]]].split('.')[0] + '.xml'),rb_dir_xml)

# 人工辅助删除剩下不清晰的图片
list_manual = [34,35,41,51,52,54,62,65,66,72,73,75,81,90,96,98,123,129,131,134,135,138,152]
print(len(list_manual))
for idx in list_manual:
    shutil.move(os.path.join(img_dir,str(idx).zfill(4) + '.jpg'),rb_dir_img)
    shutil.move(os.path.join(xml_dir,str(idx).zfill(4) + '.xml'),rb_dir_xml)

############################################## 第二步 统计每个物品图片中出现的概率 ######################################
# 将剩下的122张图片，其中102张用来训练，20张用来验证
# 统计每个测试物体出现的图片的数量，确保验证集和训练集上每个物体出现的次数相对均衡
labels_dir = os.path.join(cwd,'labels_items.txt')
f = open(labels_dir,'r')
lines = f.readlines()
f.close()
labels = []
for line in lines:
    if 'name' in line:
        labels.append(line.split("'")[1])
print(labels)


#创建每个lable的对应图片列表
dict_label_img = {}
dict_label_cnt = {}
dict_label_on = {}
for lbs in labels:
    dict_label_img[lbs] = list()
    dict_label_cnt[lbs] = 0
    dict_label_on[lbs] = False
    
xml_list = os.listdir(xml_dir)
for xmlf in xml_list:
    if '.xml' in xmlf:
        tree = etree.parse(os.path.join(xml_dir,xmlf))
        xml_root = tree.getroot()
        
        # extract xml
        for obj in xml_root:
            if(obj.tag == 'object'):
                for name in obj:
                    if(name.tag == 'name'):
                        #check in the labels
                        for lbs in labels:
                            if name.text == lbs:
                                dict_label_on[lbs] = True
        
        for lbs in labels:
            if dict_label_on[lbs] == True:
                dict_label_cnt[lbs] = dict_label_cnt[lbs] + 1
                dict_label_img[lbs].append((xmlf.split('.')[0] + '  1\n').encode())
            else:
                dict_label_img[lbs].append((xmlf.split('.')[0] + ' -1\n').encode())
            dict_label_on[lbs] = False
print(dict_label_cnt)

############################################## 第三步 分配训练集和验证集 ######################################
#从数据来看，几乎每张照片都包含了5个分类的物体，可以随机分配100张图片作为训练集，剩下的作为验证集
list_all = list(range(120))
list_val = random.sample(list_all, 20)

#write to 'trainval' text file
for lbs in labels:
    lines = []
    file = open(os.path.join(split_dir,lbs + '_trainval.txt'),'wb')
    file.writelines(dict_label_img[lbs])
    file.close()

dict_label_img_train = {}
dict_label_img_val = {}
for lbs in labels:
    dict_label_img_train[lbs] = list()
    dict_label_img_val[lbs] = list()
for sap in range(120):
    if sap in list_val:
        for lbs in labels:
            dict_label_img_val[lbs].append(dict_label_img[lbs][sap])
    else:
        for lbs in labels:
            dict_label_img_train[lbs].append(dict_label_img[lbs][sap])

#write to 'train','val' text file
for lbs in labels:
    lines = []
    file = open(os.path.join(split_dir,lbs + '_train.txt'),'wb')
    file.writelines(dict_label_img_train[lbs])
    file.close()
    
    file = open(os.path.join(split_dir,lbs + '_val.txt'),'wb')
    file.writelines(dict_label_img_val[lbs])
    file.close()

