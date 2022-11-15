#!/usr/bin/env python3
# encoding:utf-8
import cv2
import glob
import numpy as np
from CalibrationConfig import *

#开始标定, 按任意键关闭最后的图像显示

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((calibration_size[1]*calibration_size[0],3), np.float32)
objp[:,:2] = np.mgrid[0:calibration_size[1],0:calibration_size[0]].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#标定采集的图像存储路径
images = glob.glob(save_path + '*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (calibration_size[1],calibration_size[0]),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (calibration_size[1],calibration_size[0]), corners2,ret)
        cv2.imshow('img',img)
        cv2.waitKey(1)
    else:
        print('Not find object points:', fname)

cv2.destroyAllWindows()

#获取内外参
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print('mtx', mtx)
print('dist', dist)

#误差
mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error

print ("total error: ", mean_error/len(objpoints))

#保存参数
np.savez(calibration_param_path, dist_array = dist, mtx_array = mtx, fmt="%d", delimiter=" ")
print('save successful')

#读取第10张图片测试矫正
img = cv2.imread(save_path + '10.jpg')
h, w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)

cv2.imshow('calibration', dst)
cv2.imshow('original', img)
key = cv2.waitKey(0)
if key != -1:
    cv2.destroyAllWindows()
