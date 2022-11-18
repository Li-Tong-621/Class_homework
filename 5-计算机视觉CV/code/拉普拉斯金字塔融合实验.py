import cv2
import numpy as np, sys

A = cv2.imread('apple.png')
B = cv2.imread('orange.png')
print(A.shape)
print(B.shape)

# generate Gaussian pyramid for A
level = 1
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

# image with direct connecting each half
real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))

cv2.imwrite('Pyramid_blending1.jpg', ls_)
cv2.imwrite('Direct_blending.jpg', real)
