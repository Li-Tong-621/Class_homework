import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('banma.png')
img_mean3 = cv2.blur(img, (3,3))
img_mean5 =cv2.blur(img, (5,5))
img_mean7 =cv2.blur(img, (7,7))
img_mean10 =cv2.blur(img, (9,9))
img_mean100 =cv2.blur(img, (99,99))
# 展示不同的图片
titles = ['srcImg','mean—3', 'mean—5', 'mean—7', 'mean—9','mean—99']
imgs = [img, img_mean3 , img_mean5, img_mean7, img_mean10,img_mean100]
count=6
for i in range(count):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()