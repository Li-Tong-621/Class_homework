import cv2
import numpy as np, sys
def multiScaleSharpen(src, Radius):
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
    #原来在这里尝试了用直接拼接进行细节增强
    D2 = B1 - B2
    D3 = B2 - B3
    w1 = 0.5
    w2 = 0.5
    w3 = 0.25
    result = (1 - w1 * np.sign(D1)) * D1 + w2 * D2 + w3 * D3 + src
    result_img = cv2.convertScaleAbs(result)
    return result_img

A = cv2.imread('apple.png')
B = cv2.imread('orange.png')
print(A.shape)
print(B.shape)

# generate Gaussian pyramid for A
level = 4
G = A.copy()
gpA = [G]
for i in range(level):
    G = cv2.pyrDown(G)
    gpA.append(G)

# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(level):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A

lpA = [gpA[level-1]]
for i in range(level-1, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    rows, cols = gpA[i - 1].shape[ : 2]
    GE = cv2.resize(GE, (rows, cols))
    L = cv2.subtract(gpA[i - 1],GE)
    lpA.append(L)

# generate Laplacian Pyramid for B
lpB = [gpB[level-1]]
for i in range(level-1, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    rows, cols = gpB[i - 1].shape[ : 2]
    GE = cv2.resize(GE, (rows, cols))
    L = cv2.subtract(gpB[i - 1],GE)
    lpB.append(L)

# Now add left and right halves of images in each level
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:int(cols / 2)], lb[:, int(cols / 2):]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1, level):
    ls_ = cv2.pyrUp(ls_)
    rows, cols = ls_.shape[: 2]
    LS[i] = cv2.resize(LS[i], (rows, cols))
    ls_ = cv2.add(ls_, LS[i])
    multiScaleSharpen(ls_, 5)
    #每个层次细节都增强会怎么样

# image with direct connecting each half
real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))

cv2.imshow('Pyramid_blending.jpg', ls_)
cv2.waitKey(0)
