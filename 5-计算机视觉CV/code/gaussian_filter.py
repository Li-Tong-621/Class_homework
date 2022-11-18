import numpy as np
import cv2
import matplotlib.pyplot as plt
img = cv2.imread('banma.png')
img_Guassian3 = cv2.GaussianBlur(img,(3,3),0)
img_Guassian5 = cv2.GaussianBlur(img,(5,5),0)
img_Guassian7 = cv2.GaussianBlur(img,(7,7),0)

img_Guassian10 = cv2.GaussianBlur(img,(9,9),0)
img_Guassian100 = cv2.GaussianBlur(img,(99,99),0)
# 展示不同的图片
titles = ['srcImg','Guassian—3', 'Guassian—5', 'Guassian—7', 'Guassian—9','Guassian—99']
imgs = [img, img_Guassian3 , img_Guassian5, img_Guassian7, img_Guassian10,img_Guassian100]
count=6
for i in range(count):
    plt.subplot(2,3,i+1)#注意，这和matlab中类似，没有0，数组下标从1开始
    plt.imshow(imgs[i])
    plt.title(titles[i])
plt.show()