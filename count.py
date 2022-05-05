import os
import json

import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob

pic_path = './case1/'
pic_name= 'Snap-732-Draw Scale Bar Annotation-16-Image Export-01.jpg'
json_path = glob(f'case1/*.json')[0]
with open(json_path, 'r') as f:
    json_dict = json.load(f)
regions = json_dict[pic_name]['regions']


def apply_mask(pic,mas):
    return 255 - cv2.add(pic, 255-mas)



img = cv2.imread(pic_path+pic_name,1)

mask = np.zeros(img.shape[:2], dtype='uint8')
for region in regions:
    x_locs = np.array([region['shape_attributes']['all_points_x']])
    y_locs = np.array([region['shape_attributes']['all_points_y']])
    locs = np.concatenate((x_locs.T, y_locs.T), axis=1)
    pts = np.array([locs], dtype=np.int32)

    # 生成 mask
    mask = np.zeros(img.shape[:2], dtype='uint8')
    cv2.polylines(mask, pts, 1, 255)  # 绘制多边形
    cv2.fillPoly(mask, pts, 255)  # 填充


def countcell(img,mask):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(otsu(gray))

    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
    mask = 255 - mask
    visualization = 255 - cv2.add(thresh, mask)
    kernel = np.ones((2, 2), np.uint8)  # 进行腐蚀膨胀操作
    erosion = cv2.erode(visualization, kernel, iterations=3)
    dilation = cv2.dilate(erosion, kernel, iterations=3)
    contours, hirearchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找出连通域
    # 对连通域面积进行比较
    area = []  # 建立空数组，放连通域面积
    contours1 = []  # 建立空数组，放减去后的数组
    for i in contours:
        # area.append(cv2.contourArea(i))
        # print(area)
        if cv2.contourArea(i) > 10:  # 计算面积 去除面积小的 连通域
            contours1.append(i)
    return (len(contours1) - 1)  # 计算连通域个数

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(otsu(gray))

ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
thresh = cv2.GaussianBlur(thresh, (3, 3), 0)
mask = 255 - mask
visualization = 255 - cv2.add(thresh, mask)
kernel = np.ones((2, 2), np.uint8)  # 进行腐蚀膨胀操作
erosion = cv2.erode(visualization, kernel, iterations=3)
dilation = cv2.dilate(erosion, kernel, iterations=3)
contours, hirearchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 找出连通域
    # 对连通域面积进行比较
area = []  # 建立空数组，放连通域面积
contours1 = []  # 建立空数组，放减去后的数组
for i in contours:
        # area.append(cv2.contourArea(i))
        # print(area)
    if cv2.contourArea(i) > 10:  # 计算面积 去除面积小的 连通域
        contours1.append(i)


print(countcell(img,mask))
cv2.namedWindow("display",0);
cv2.resizeWindow("display", 1200, 800);
cv2.imshow("display",thresh)

cv2.waitKey()