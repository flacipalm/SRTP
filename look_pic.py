import cv2

import json
import os
import numpy as np
import cv2
import random

# 关节点连接性，COCO数据集关节点标注文件最后一行有提供
BODY_PARTS = [
    (16, 14),
    (14, 12),
    (17, 15),
    (15, 13),
    (12, 13),
    (6, 12),
    (7, 13),
    (6, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (9, 11),
    (2, 3),
    (1, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (5, 7)
]

parent_id = '/home/gaoyao/dataset/coco/2017/images/val2017'
imgid = 872
image_id = os.path.join(parent_id, str(imgid).zfill(12) + '.jpg')
anno_file = 'keypoints_val2017_results.json'
with open(anno_file) as annotations:
    annos = json.load(annotations)
colors = [(204, 204, 0), (51, 153, 51), (51, 153, 255)]  # 标注关节点的颜色


def showAnns(img, keypoints, BODY_PARTS):
    img = img.copy()
    for i in range(len(keypoints)):
        kpt = np.array(keypoints[i]).reshape(-1, 3)

        for j in range(kpt.shape[0]):
            x = kpt[j][0]
            y = kpt[j][1]
            cv2.circle(img, (int(x), int(y)), 8, colors[i], -1)  # 画点

        for part in BODY_PARTS:
            # 通过body_part_m来连线
            # 部位为part[0]的节点坐标，这里需要减去1.是因为得到的结果是以0开头的，而提供的是以1开头
            keypoint_1 = kpt[part[0] - 1]
            x_1 = int(keypoint_1[0])  # 单个部位坐标x
            y_1 = int(keypoint_1[1])
            keypoint_2 = kpt[part[1] - 1]
            x_2 = int(keypoint_2[0])
            y_2 = int(keypoint_2[1])
            if keypoint_1[2] > 0 and keypoint_2[2] > 0:
                # 画线  参数--# img:图像，起点坐标，终点坐标，颜色，线的宽度
                # opencv读取图片通道为BGR
                cv2.line(img, (x_1, y_1), (x_2, y_2), colors[i], 3)
    cv2.imshow('keypoints', img)
    cv2.waitKey(20000)


kpts = []
for i in range(len(annos)):
    # import pdb; pdb.set_trace()
    if imgid != annos[i]['image_id']:
        continue
    keypoints = annos[i]['keypoints']
    kpts.append(keypoints)
img = cv2.imread(image_id)
showAnns(img, kpts, BODY_PARTS)



a = cv2.imread("test.jpg")
cv2.imwrite("/home/xqr/fzy/res.jpg", a)