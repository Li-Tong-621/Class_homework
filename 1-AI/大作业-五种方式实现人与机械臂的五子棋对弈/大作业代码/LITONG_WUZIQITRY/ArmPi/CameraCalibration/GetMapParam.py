#!/usr/bin/env python3
# encoding:utf-8
import cv2
import time
import numpy as np
from CalibrationConfig import *

#获取像素与实际距离的映射系数, 按space键获取参数，按其他任意键退出
#注意:获取参数需要摄像头画面能完整的看到整个棋盘，且十字正对棋盘

cap = cv2.VideoCapture(-1)

#加载参数
param_data = np.load(calibration_param_path + '.npz')

#获取参数
mtx = param_data['mtx_array']
dist = param_data['dist_array']

while True:
    ret, frame = cap.read()
    if ret:
        h, w = frame.shape[:2]
        break
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w,h), 5)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ret, Frame = cap.read()
    if ret:
        frame = Frame.copy()
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        img = dst.copy()

        cv2.line(dst, (0, int(h / 2)), (w, int(h / 2)), (0, 0, 255), 2)
        cv2.line(dst, (int(w / 2), 0), (int(w / 2), h), (0, 0, 255), 2)        
        cv2.imshow('dst',dst)
        key = cv2.waitKey(1)
        if key == 32:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret_, corners = cv2.findChessboardCorners(gray, (calibration_size[1], calibration_size[0]),None)
            if ret_:
                corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                sum_ = []
                last_i = [0]
                count = 0
                for i in corners2:
                    count += 1
                    if count != 1 and (count - 1)%7 != 0:
                        a_ = (last_i[0] - i[0])**2    
                        sum_.append(np.sqrt(np.sum(a_)))
                    last_i = i
                
                map_param = np.mean(sum_)
                map_param = corners_length/map_param

                np.savez(map_param_path, map_param = map_param, fmd='%d', delimiter=' ')
                print('save successful')
        if key == 27:
            break
    else:
        time.sleep(0.01)
cap.release()
cv2.destroyAllWindows()
