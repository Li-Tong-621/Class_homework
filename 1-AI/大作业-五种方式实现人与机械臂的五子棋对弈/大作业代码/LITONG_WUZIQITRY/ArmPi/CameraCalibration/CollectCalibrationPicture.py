#!/usr/bin/env python3
# encoding:utf-8
import os
import cv2
from CalibrationConfig import *

#采集标定图像，保存在calib文件夹下
#按下键盘上的space键存储图像，按esc退出

cap = cv2.VideoCapture(-1)

#如果calib文件夹不存在，则新建
if not os.path.exists(save_path):
    os.mkdir(save_path)

#计算存储的图片数量
num = 0
while True:
    ret, frame = cap.read()
    if ret:
        Frame = frame.copy()
        cv2.putText(Frame, str(num), (10, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (0, 0, 255), 5)
        cv2.imshow("Frame", Frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == 32:
            num += 1
            #图片名称格式：当前图片数量.jpg
            cv2.imwrite(save_path + str(num) + ".jpg", frame) 

cap.release()
cv2.destroyAllWindows()
