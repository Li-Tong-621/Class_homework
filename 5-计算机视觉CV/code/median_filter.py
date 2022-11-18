import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('banma.png')
img_median3 = cv2.medianBlur(img, 3)
img_median5 =cv2.medianBlur(img, 5)
img_median7 =cv2.medianBlur(img, 7)
img_median10 =cv2.medianBlur(img, 9)
img_median100 =cv2.medianBlur(img, 99)
# 展示不同的图片
titles = ['srcImg','median—3', 'median—5', 'median—7', 'median—9','median—99']
imgs = [img, img_median3 , img_median5, img_median7, img_median10,img_median100]
count=6
for i in range(count):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()
