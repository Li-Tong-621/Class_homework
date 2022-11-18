import cv2
import glob
import os
import numpy as np


def multiScaleSharpen(src,img2, Radius):
    sigma1 = 1.0
    sigma2 = 2.0
    sigma3 = 4.0
    B1 = cv2.GaussianBlur(src, (Radius, Radius), sigma1)
    B2 = cv2.GaussianBlur(src, (Radius * 2 - 1, Radius * 2 - 1), sigma2)
    B3 = cv2.GaussianBlur(src, (Radius * 4 - 1, Radius * 4 - 1), sigma3)
    src = src.astype(np.float)  # uint8 to float, avoid Saturation
    B1 = B1.astype(np.float)
    B2 = B2.astype(np.float)
    B3 = B3.astype(np.float)
    D1 = src - B1  # get detail
    #D1 = img2  - B1 # get detail
    D2 = B1 - B2
    D3 = B2 - B3
    w1 = 0.5
    w2 = 0.5
    w3 = 0.25
    result = (1 - w1 * np.sign(D1)) * D1 + w2 * D2 + w3 * D3 + img2
    result_img = cv2.convertScaleAbs(result)
    return result_img


if __name__ == "__main__":
    img = cv2.imread("Pyramid_blending5.jpg")
    img2=cv2.imread("Direct_blending.jpg")

    import cv2
    import numpy as np, sys

    """A = cv2.imread('apple.png')
    B = cv2.imread('orange.png')
    rows, cols = A.shape[: 2]
    real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))
    for i in range(1, int(rows / 32)):
        # L=real[:int(rows/2)-i,:]
        for j in range(35):
            x = np.random.normal(loc=i, scale=2, size=None)
            x = int(x)
            x = abs(x)
            # M=real[int(rows / 2)-x:int(rows / 2)+x,int(rows / 2)-i:int(rows / 2)+i]
            M = real[:, int(rows / 2) - x - i:int(rows / 2) + x + i]
            # R=real[int(rows / 2)+i:,:]
            # M=cv2.cv2.blur(M, (i,i))
            M = cv2.GaussianBlur(M, (i % 2 * 2 + 1, i % 2 * 2 + 1), 0)
            real[:, int(rows / 2) - x - i:int(rows / 2) + x + i] = M
"""
    result = multiScaleSharpen(img,img, 5)

    cv2.imshow("1", result)
    cv2.imwrite('Pyramid_blending5_ditail.jpg',result)
    cv2.waitKey(0)