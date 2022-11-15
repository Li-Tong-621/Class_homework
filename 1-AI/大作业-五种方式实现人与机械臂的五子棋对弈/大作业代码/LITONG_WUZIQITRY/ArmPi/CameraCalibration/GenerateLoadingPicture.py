#!/usr/bin/env python3
# encoding:utf-8
import cv2
import numpy as np

#生成加载图, 按键盘任意键退出

#分辨率
size = (480, 640)

loading_picture = np.zeros(size)
loading_picture[:] = 255
cv2.putText(loading_picture, 'Loading......', (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite("loading.jpg", loading_picture)
cv2.imshow("loading_picture", loading_picture)
key = cv2.waitKey(0)
if key != -1:
    cv2.destroyAllWindows()
