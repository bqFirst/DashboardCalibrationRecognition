# -*- coding: utf-8 -*-
# @Time    : 2021/5/21 11:13
# @Author  : Davion.W
# @Email   : 1617225808@qq.com
# @File    : main.py

"""
1、采用run.py 方法获取0刻度，计算指针与0刻角度
2、采用MeterReader.py 方法获取圆
3、采用 仪表数据读取.py 方法获取指针在图像坐标的旋转角度
"""
from MarkZero import markzero
from MeterReader import METER
from 模板匹配法 import get_match_rect, v2_by_k_means, get_pointer_rad, get_rad_val
from angle import angle

import cv2

# 1、 获取0刻度
template_file = 'template.png'
template1_file = 'template1.png'
file = 'test/(1).jpg'

# template_file = '5.jpg'
# template1_file = '5.jpg'
# file = '5.jpg'

view_show = True
# template1_file = template1_file= 'meter/meter-0001.jpg'
# file = 'meter/meter-0001.jpg'

# template1_file = template_file = '5.jpg'
# file = '5.jpg'


# 读取模板
img_s = cv2.imread(file)
img = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)
template = cv2.imread(template1_file)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# 通过模板匹配分割仪表盘，返回仪表盘坐标
top_left, bottom_right = get_match_rect(template, img, method=cv2.TM_CCOEFF)

# 可视化，绘制仪表盘矩形
if view_show:
    cv2.rectangle(img_s, top_left, bottom_right, 255, 2)
    cv2.imshow('meter', cv2.resize(img_s,(int(img.shape[1]*0.5), int(img.shape[0]*0.5))))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

#######################################################
# 分割仪表盘图像，并将分割的仪表盘进行第二次模板匹配,获得更精准仪表盘
new = img_s[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]
template = cv2.imread(template_file)
top_left, bottom_right = get_match_rect(template, new, method=cv2.TM_CCOEFF)
new_ = new[top_left[1]:bottom_right[1] + 1, top_left[0]:bottom_right[0] + 1]

if view_show:
    cv2.imshow('meter2', new_)
    # cv2.waitKey(0)


# 2、获取圆、圆心
cir_img, circles = METER("", new_).iden_pic()
c_x, c_y, r = circles[0, 0, 0], circles[0, 0, 1], circles[0, 0, 2]
# print('c_x, c_y, r ', circles[0, 0, 0], circles[0, 0, 1], circles[0, 0, 2])


if view_show:
    cv2.imshow('circle', cir_img)
    # cv2.waitKey(0)

# kmeans 二值化
img = v2_by_k_means(new_, circles)

if view_show:
    cv2.imshow('kmeans ', img)
    cv2.waitKey(0)

# 指针拟合
rad = get_pointer_rad(img, circles)  # 重合数量、坐标
l, x, y = rad[0], rad[1][0], rad[1][1]
print('l, x, y', l, x, y)
# 鼠标获得0刻度
opint = markzero(new_)
# print("0刻度坐标 ", opint)

rad = angle((c_x, c_y, opint[0], opint[1]), (c_x, c_y, x, y))
print('夹角 ', rad)

font = cv2.FONT_HERSHEY_SIMPLEX
imgzi = cv2.putText(new_, str(rad), (c_x-20, c_y+20), font, 1.2, (255, 255, 255), 1)

cv2.line(new_, (c_x, c_y), (int(x), int(y)), (0, 255, 0), thickness=1)
cv2.line(new_, (c_x, c_y), (int(opint[0]), int(opint[1])), (0, 255, 0), thickness=1)
cv2.imshow('rad ', new_)
cv2.waitKey(0)


# print('rad ', rad)
# print('rad[1] ', rad[1])
# rad_val = get_rad_val(rad[1])
# print('当前角度 ', rad_val)
cv2.waitKey(0)
cv2.destroyAllWindows()

