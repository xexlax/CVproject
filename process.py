import os
import json
import numpy as np
import cv2
from glob import glob

case_path = './case1'                       # case
mask_path = './masks'                       # 掩码生成文件夹
visualization_path = './visualization'      # 可视化结果文件夹

img_path_arr = glob(f'{case_path}/*.jpg')
json_path = glob(f'{case_path}/*.json')[0]

if not os.path.exists(mask_path):
    os.makedirs(mask_path)

if not os.path.exists(visualization_path):
    os.makedirs(visualization_path)

# 解析 json 为 python dict
with open(json_path, 'r') as f:
    json_dict = json.load(f)

# 对每一张图片，提取离区掩码并生成可视化结果
for img_path in img_path_arr:
    # Windows 下 glob.glob 最后以 '\\' 分隔。MacOS / Linux 可能需要修改为 '/'
    file_name = img_path.split('\\')[-1]
    regions = json_dict[file_name]['regions']
    img = cv2.imread(img_path)

    # 对每个标注的离区
    for region in regions:
        x_locs = np.array([region['shape_attributes']['all_points_x']])
        y_locs = np.array([region['shape_attributes']['all_points_y']])
        locs = np.concatenate((x_locs.T, y_locs.T), axis=1)
        pts = np.array([locs], dtype=np.int32)

        # 生成 mask
        mask = np.zeros(img.shape[:2], dtype='uint8')
        cv2.polylines(mask, pts, 1, 255)    # 绘制多边形
        cv2.fillPoly(mask, pts, 255)        # 填充

        # 生成可视化结果
        mask_color = np.zeros(img.shape, dtype='uint8')
        mask_color[mask == 255] = (0, 0, 255)
        visualization = cv2.add(img, mask_color)

        # 写入
        cv2.imwrite(f'{mask_path}/mask_{file_name}', mask)
        cv2.imwrite(
            f'{visualization_path}/visualization_{file_name}', visualization)
    print(file_name)
