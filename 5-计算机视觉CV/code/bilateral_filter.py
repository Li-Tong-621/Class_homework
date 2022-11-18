import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('banma.png')

"""img_bilateral3 = cv2.bilateralFilter(img,6,75,75)
img_bilateral5 =cv2.bilateralFilter(img,10,75,75)
img_bilateral7 =cv2.bilateralFilter(img,14,75,75)
img_bilateral10 =cv2.bilateralFilter(img,18,75,75)
img_bilateral100 =cv2.bilateralFilter(img,198,75,75)
# 展示不同的图片
titles = ['srcImg','bilateral—3', 'bilateral—5', 'bilateral—7', 'bilateral—9','bilateral—99']
imgs = [img, img_bilateral3 , img_bilateral5, img_bilateral7, img_bilateral10,img_bilateral100]
count=6
for i in range(count):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()
"""
k=['198,0,0)','198,75,0)','198,0,75)','198,150,0)','198,0,150)','198,200,0)','198,0,200)']
k2=['18,75,75)']
for i in k:
    x='cv2.bilateralFilter(img,'+i
    print(i)
    print(x)
    y=eval(x)
    cv2.imshow('bilateral',y)
    cv2.waitKey(0)