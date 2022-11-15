#!/usr/bin/env python3
# encoding:utf-8
import cv2
import numpy as np
from calibration_config import *

#生成标定棋盘, 按键盘任意键退出

#棋盘分辨率
size = (640, 640)

calibration_board = np.zeros(size)
block_width = size[0]//(calibration_size[1] + 1)
black_block = np.full((block_width, block_width), 255)

for row in range((calibration_size[0] + 1)):
    for col in range((calibration_size[1] + 1)):
        if (row+col)%2==0:
            row_begin = row*block_width
            row_end = row_begin + block_width
            col_begin = col*block_width
            col_end = col_begin + block_width
            calibration_board[row_begin:row_end, col_begin:col_end] = black_block

cv2.imwrite("calibration_board.jpg", calibration_board)
cv2.imshow("calibration_board", calibration_board)
key = cv2.waitKey(0)
if key != -1:
    cv2.destroyAllWindows()
