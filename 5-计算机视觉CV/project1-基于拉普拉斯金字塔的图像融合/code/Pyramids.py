import cv2 as cv
import numpy as np

"""
def laplaian_demo(pyramid_images):
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):
        if (i-1) < 0:
            h, w = src.shape[:2]
            expand = cv.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = cv.subtract(src, expand)# + 127
            cv.imshow("lpls_" + str(i), lpls)
        else:
            h, w = pyramid_images[i-1].shape[:2]
            expand = cv.pyrUp(pyramid_images[i], dstsize=(w, h))
            lpls = cv.subtract(pyramid_images[i-1], expand) #+ 127
            cv.imshow("lpls_"+str(i), lpls)


def pyramid_up(image, level=5):
    temp = image.copy()
    # cv.imshow("input", image)
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        # cv.imshow("pyramid_up_" + str(i), dst)
        temp = dst.copy()
    return pyramid_images


src = cv.imread("lena.png")
cv.namedWindow("input", cv.WINDOW_AUTOSIZE)
cv.imshow("input", src)
# pyramid_up(src)
laplaian_demo(pyramid_up(src))

cv.waitKey(0)
cv.destroyAllWindows()"""

import cv2

"""img = cv2.imread('lena.png')
row, col, _ = img.shape
print(int(row/2), int(col/2))
lower_reso1 = cv2.pyrDown(img)
lower_reso2 = cv2.pyrDown(lower_reso1)
lower_reso3 = cv2.pyrDown(lower_reso2)
cv2.imshow('origin image', img)
cv2.imshow('pyrDown./2', lower_reso1)
cv2.imshow('pyrDown./4', lower_reso2)
cv2.imshow('pyrDown./8', lower_reso3)

higher_reso3 = cv2.pyrUp(lower_reso3)
higher_reso2 = cv2.pyrUp(higher_reso3)
higher_reso1 = cv2.pyrUp(higher_reso2)
cv2.imshow('pyrUp*2', higher_reso3)
cv2.imshow('pyrUp*4', higher_reso2)
cv2.imshow('pyrUp*8', higher_reso1)
cv2.waitKey(0)"""
"""def gaussian(ori_image, down_times=5):
    # 1：添加第一个图像为原始图像
    temp_gau = ori_image.copy()
    gaussian_pyramid = [temp_gau]
    for i in range(down_times):
        # 2：连续存储5次下采样，这样高斯金字塔就有6层
        temp_gau = cv2.pyrDown(temp_gau)
        gaussian_pyramid.append(temp_gau)
    return gaussian_pyramid


def laplacian(gaussian_pyramid, up_times=5):
    laplacian_pyramid = [gaussian_pyramid[-1]]

    for i in range(up_times, 0, -1):
        # i的取值为5,4,3,2,1,0也就是拉普拉斯金字塔有6层
        temp_pyrUp = cv2.pyrUp(gaussian_pyramid[i])
        rows, cols = gaussian_pyramid[i - 1].shape[: 2]
        temp_pyrUp = cv2.resize(temp_pyrUp, (rows, cols))

        temp_lap = cv2.subtract(gaussian_pyramid[i -1], temp_pyrUp)
        laplacian_pyramid.append(temp_lap)
    return laplacian_pyramid

img = cv2.imread('lena.png')
x=gaussian(img)
cv2.imshow('0',x[0])
cv2.imshow('1',x[1])
cv2.imshow('2',x[2])
cv2.imshow('3',x[3])
cv2.imshow('4',x[4])
cv2.imshow('5',x[5])
#cv2.waitKey(0)
y=laplacian(img)
cv2.imshow('0',y[0])
cv2.imshow('1',y[1])
cv2.imshow('2',y[2])
cv2.imshow('3',y[3])
cv2.imshow('4',y[4])
cv2.imshow('5',y[5])
cv2.waitKey(0)"""

def pyramid_demo(image):
    level = 4 #自定义lever
    temp = image.copy()
    pyramid_images = []

    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid_down_"+str(i+1), dst)
        cv2.imwrite("pyramid_down_"+str(i+1)+'.png', dst)

        temp = dst.copy()
    return pyramid_images
#_________0 最大的，越往下越小，因为是不断地append
def laplace_demo(image):
    pyramid_images = pyramid_demo(image)
    level = len(pyramid_images)
 # ————————————————(start, stop [,step])，也就是倒着 从最小的开始
    for i in range(level-1, -1, -1):
        if i-1 < 0:#如果是最大的,单独拿出来主要是因为原图image没有在里面。
            expand  = cv.pyrUp(pyramid_images[i], dstsize=image.shape[:2])
            lpls = cv.subtract(image, expand)
            cv.imshow("laplace_demo"+str(i), lpls)
            cv2.imwrite("laplace_demo"+str(i)+'.png', lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            if i==1:
                print(1)
                cv2.imwrite('!' + str(i) + '.png', expand)
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("laplace_demo"+str(i), lpls)
            cv2.imwrite("laplace_demo" + str(i)+'.png', lpls)
#————————————————(start, stop [,step])
img = cv2.imread('lena.png')
pyramid_demo(img)
laplace_demo(img)
cv2.waitKey(0)