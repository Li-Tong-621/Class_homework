#!/usr/bin/env python3
# encoding:utf-8
import cv2
import sys
sys.path.append('/home/pi/ArmPi/')
import math
import numpy as np
from CameraCalibration.CalibrationConfig import *

#机械臂原点即云台中心，距离摄像头画面中心的距离， 单位cm
image_center_distance = 20

#加载参数
param_data = np.load(map_param_path + '.npz')

#计算每个像素对应的实际距离
map_param_ = param_data['map_param']

#数值映射
#将一个数从一个范围映射到另一个范围
def leMap(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

#将图形的像素坐标转换为机械臂的坐标系
#传入坐标及图像分辨率，例如(100, 100, (640, 320))
def convertCoordinate(x, y, size):
    x = leMap(x, 0, size[0], 0, 640)
    x = x - 320
    x_ = round(x * map_param_, 2)

    y = leMap(y, 0, size[1], 0, 480)
    y = 240 - y
    y_ = round(y * map_param_ + image_center_distance, 2)

    return x_, y_

#将现实世界的长度转换为图像像素长度
#传入坐标及图像分辨率，例如(10, (640, 320))
def world2pixel(l, size):
    l_ = round(l/map_param_, 2)

    l_ = leMap(l_, 0, 640, 0, size[0])

    return l_

# 获取检测物体的roi区域
# 传入cv2.boxPoints(rect)返回的四个顶点的值，返回极值点
def getROI(box):
    x_min = min(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
    x_max = max(box[0, 0], box[1, 0], box[2, 0], box[3, 0])
    y_min = min(box[0, 1], box[1, 1], box[2, 1], box[3, 1])
    y_max = max(box[0, 1], box[1, 1], box[2, 1], box[3, 1])

    return (x_min, x_max, y_min, y_max)

#除roi区域外全部变成黑色
#传入图形，roi区域，图形分辨率
def getMaskROI(frame, roi, size):
    x_min, x_max, y_min, y_max = roi
    x_min -= 10
    x_max += 10
    y_min -= 10
    y_max += 10

    if x_min < 0:
        x_min = 0
    if x_max > size[0]:
        x_max = size[0]
    if y_min < 0:
        y_min = 0
    if y_max > size[1]:
        y_max = size[1]

    black_img = np.zeros([size[1], size[0]], dtype=np.uint8)
    black_img = cv2.cvtColor(black_img, cv2.COLOR_GRAY2RGB)
    black_img[y_min:y_max, x_min:x_max] = frame[y_min:y_max, x_min:x_max]
    
    return black_img

# 获取木块中心坐标
# 传入minAreaRect函数返回的rect对象， 木快极值点， 图像分辨率， 木块边长
def getCenter(rect, roi, size, square_length):
    x_min, x_max, y_min, y_max = roi
    #根据木块中心的坐标，来选取最靠近图像中心的顶点，作为计算准确中心的基准
    if rect[0][0] >= size[0]/2:
        x = x_max 
    else:
        x = x_min
    if rect[0][1] >= size[1]/2:
        y = y_max
    else:
        y = y_min

    #计算木块的对角线长度
    square_l = square_length/math.cos(math.pi/4)

    #将长度转换为像素长度
    square_l = world2pixel(square_l, size)

    #根据木块的旋转角来计算中心点
    dx = abs(math.cos(math.radians(45 - abs(rect[2]))))
    dy = abs(math.sin(math.radians(45 + abs(rect[2]))))
    if rect[0][0] >= size[0] / 2:
        x = round(x - (square_l/2) * dx, 2)
    else:
        x = round(x + (square_l/2) * dx, 2)
    if rect[0][1] >= size[1] / 2:
        y = round(y - (square_l/2) * dy, 2)
    else:
        y = round(y + (square_l/2) * dy, 2)

    return  x, y

# 获取旋转的角度
# 参数：机械臂末端坐标, 木块旋转角
def getAngle(x, y, angle):
    theta6 = round(math.degrees(math.atan2(abs(x), abs(y))), 1)
    angle = abs(angle)
    
    if x < 0:
        if y < 0:
            angle1 = -(90 + theta6 - angle)
        else:
            angle1 = theta6 - angle
    else:
        if y < 0:
            angle1 = theta6 + angle
        else:
            angle1 = 90 - theta6 - angle

    if angle1 > 0:
        angle2 = angle1 - 90
    else:
        angle2 = angle1 + 90

    if abs(angle1) < abs(angle2):
        servo_angle = int(500 + round(angle1 * 1000 / 240))
    else:
        servo_angle = int(500 + round(angle2 * 1000 / 240))
    return servo_angle
