import os

import numpy as np
import cv2
from glob import glob


def Segmentation(pic_path, pic_name, predict_path, i):
    # 读取图片
    img = cv2.imread(pic_path + pic_name, 1)

    # 转化为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
    visualization = cv2.GaussianBlur(thresh, (3, 3), 0)

    # 先对未经过腐蚀膨胀的原图进行连通域的筛选，将部分切片细胞重叠的区域和杂质去除
    contours, hirearchy = cv2.findContours(visualization, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选出面积大于200的连通域，认为该部分连通域为重叠区域（大型色块）；小于10的区域为杂质或噪点
    contours2 = []
    for k in contours:
        if cv2.contourArea(k) > 200 or cv2.contourArea(k) < 10:
            contours2.append(k)

    # 将对应的区域扣除
    visualization1 = 255 - visualization
    draw2 = cv2.drawContours(visualization1, contours2, -1, (0, 0, 0), -1)
    visualization = cv2.add(visualization, draw2)

    # 定义腐蚀膨胀的kernel
    kernel = np.ones((3, 3), np.uint8)
    kernel[0][0] = kernel[0][2] = kernel[2][0] = kernel[2][2] = 0
    # 进行腐蚀膨胀操作
    erosion = cv2.erode(visualization, kernel, iterations=12)
    dilation = cv2.dilate(erosion, kernel, iterations=3)

    # 对腐蚀膨胀后的图片进行处理，找出非聚集的区域
    contours1, hirearchy1 = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours3 = []
    # 筛选出合适的面积并绘制
    for s in contours1:
        if cv2.contourArea(s) > 4000:
            contours3.append(s)
    draw = cv2.drawContours(img, contours3, -1, (255, 255, 255), -1)

    # 绘制出识别区域并叠加至原始图像
    mask = np.ones(img.shape[:3], dtype='uint8')
    mask[:] = (0, 120, 0)
    draw = cv2.drawContours(mask, contours3, -1, (0, 0, 0), -1)
    origin = cv2.imread(pic_path + pic_name, 1)
    final = cv2.add(origin, draw)
    # final为最终叠加后的图像结果
    cv2.namedWindow("final", 0);
    cv2.resizeWindow("final", 1200, 800);
    cv2.imshow("final", final)

    # final_mask为黑白图像
    final_mask = np.ones(img.shape[:2], dtype='uint8') * 255
    final_mask = cv2.drawContours(final_mask, contours3, -1, (0, 0, 0), -1)
    print("生成分割结果" + str(i+1))
    cv2.imwrite(f'{predict_path}/mask_{i + 1}.jpg', final_mask)

    cv2.namedWindow("mask", 0);
    cv2.resizeWindow("mask", 1200, 800);
    cv2.imshow("mask", final_mask)
    cv2.waitKey()


if __name__ == '__main__':
    pic_path = './case1/'
    predict_path = './predict'

    caseList = os.listdir(pic_path)
    pic_num = len(caseList) - 1
    for i in range(pic_num):
        Segmentation(pic_path, caseList[i], predict_path, i)
