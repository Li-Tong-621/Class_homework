import numpy as np
import cv2
import matplotlib.pyplot as plt
########     四个不同的滤波器    #########
img = cv2.imread('banma.png')
# 均值滤波
img_mean = cv2.blur(img, (5,5))
# 高斯滤波
img_Guassian = cv2.GaussianBlur(img,(5,5),0)
# 中值滤波
img_median = cv2.medianBlur(img, 5)
# 双边滤波
img_bilater = cv2.bilateralFilter(img,9,75,75)
# 展示不同的图片
titles = ['srcImg_原图','mean_均值滤波', 'Gaussian_高斯滤波', 'median_中值滤波', 'bilateral_双边滤波']
imgs = [img, img_mean, img_Guassian, img_median, img_bilater]

for i in range(5):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
plt.show()

