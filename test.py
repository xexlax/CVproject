import os
import json

import numpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from skimage import morphology

pic_path = './case1/'
pic_name= 'Snap-735-Draw Scale Bar Annotation-01-Image Export-04.jpg'
json_path = glob(f'case1/*.json')[0]
with open(json_path, 'r') as f:
    json_dict = json.load(f)
regions = json_dict[pic_name]['regions']





img = cv2.imread(pic_path+pic_name,1)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray=cv2.GaussianBlur(gray,(3,3),0)
thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,121,3)

thresh=255-thresh

kernel = np.ones((2, 2), np.uint8)  # 进行腐蚀膨胀操作

thresh = cv2.dilate(thresh, kernel, iterations=3)

contours1, hirearchy1 = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours3 = []
for i in contours1:
    if cv2.contourArea(i) >100000 :
        contours3.append(i)
mask = np.ones(img.shape[:3], dtype='uint8')
mask[:] = (0, 0, 0)
mask0 = cv2.drawContours(mask, contours3, -1, (255, 255, 255), -1)

maskgray=cv2.cvtColor(mask0,cv2.COLOR_BGR2GRAY)
kernel = np.ones((9, 9), np.uint8)  # 进行腐蚀膨胀操作

maskgray=cv2.erode(maskgray, kernel, iterations=40)

kernel = np.ones((3, 3), np.uint8)  # 进行腐蚀膨胀操作

mask01=maskgray

mask01[mask01==255] = 1
skeleton0 = morphology.skeletonize(mask01)   # 骨架提取

skeleton = skeleton0.astype(np.uint8)*255




# dst = cv2.cornerHarris(mask01, 5, 7, 0.2)
# mask0[dst>0.1*dst.max()]=[0,0,255]
# corners = cv2.goodFeaturesToTrack(mask01, 80, 0.2, 10)
#
# for pt in corners:
#          print(pt)
#          b = np.random.random_integers(0, 256)
#          g = np.random.random_integers(0, 256)
#          r = np.random.random_integers(0, 256)
#          x = np.int32(pt[0][0])
#          y = np.int32(pt[0][1])
#          cv2.circle(mask0, (x, y), 5, (int(b), int(g), int(r)), 2)

cv2.namedWindow("display",0);
cv2.resizeWindow("display", 1200, 800)
cv2.imshow("display",skeleton)
cv2.waitKey()