
# Gamma correction and the Power Law Transform，伽马校正也称幂律交换；使图像变得更亮或者更暗的方法；
# USAGE
# python adjust_gamma.py --image images/_L3A4387.jpg

# 导入必要的包
import numpy as np
import math
import cv2
#色彩恢复实验
def gamma_trans(img, gamma):  # gamma函数处理
    gamma_table = [np.power(x / 255.0, gamma) * 255.0 for x in range(256)]  # 建立映射表
    gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)  # 颜色值为整数
    return cv2.LUT(img, gamma_table)  # 图片颜色查表。另外可以根据光强（颜色）均匀化原则设计自适应算法。


def nothing(x):
    pass


file_path = "Pyramid_blending5_ditail.jpg"
img_gray = cv2.imread(file_path, 0)  # 灰度图读取，用于计算gamma值
img = cv2.imread(file_path)  # 原图读取

mean = np.mean(img_gray)
gamma_val = math.log10(0.5) / math.log10(mean / 255)  # 公式计算gamma

image_gamma_correct = gamma_trans(img, gamma_val)  # gamma变换

# print(mean,np.mean(image_gamma_correct))

cv2.imshow('image_raw', img)
cv2.imshow('image_gamma', image_gamma_correct)
cv2.imwrite('Pyramid_blending5_color.jpg',image_gamma_correct)
cv2.waitKey(0)
"""img = cv2.imread('Direct_blending.jpg')  # 原图读取
img_bilateral = cv2.GaussianBlur(img,(9,9),0)
cv2.imshow('image_gamma', img_bilateral)
#cv2.imwrite('image_gamma_bilateral.jpg',img_bilateral)
cv2.waitKey(0)"""

"""
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0/gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** invGamma) * 255)
    table = np.array(table).astype("uint8")
    return cv2.LUT(img_dark, table)
 
img_brighter = adjust_gamma(img_dark, 2)
"""