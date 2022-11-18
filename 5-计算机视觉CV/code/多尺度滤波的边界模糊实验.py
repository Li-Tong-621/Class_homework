import cv2
import numpy as np, sys

A = cv2.imread('apple.png')
B = cv2.imread('orange.png')
rows, cols = A.shape[: 2]
real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))
cv2.imshow('Direct_blending.jpg', real)
cv2.waitKey(0)
for i in range(1,int(cols/2)):
    L=real[:,:int(cols/2)-i]
    M=real[:,int(cols / 2)-i:int(cols / 2)+i]
    R=real[:,int(cols / 2)+i:]
    """cv2.imshow('L',L)
    cv2.imshow('M',M)
    cv2.imshow('R',R)"""
    M=cv2.GaussianBlur(M,(i%2*2+1,i%2*2+1),0)
    real=np.hstack((L,M))
    real=np.hstack((real,R))
real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))
for i in range(1,int(rows/32)):
    #L=real[:int(rows/2)-i,:]
    for j in range(35):

        x= np.random.normal(loc=i, scale=2, size=None)
        x=int(x)
        x=abs(x)
        #M=real[int(rows / 2)-x:int(rows / 2)+x,int(rows / 2)-i:int(rows / 2)+i]
        M = real[:, int(rows / 2)-x - i:int(rows / 2) + x+i]
        #R=real[int(rows / 2)+i:,:]
        #M=cv2.cv2.blur(M, (i,i))
        M = cv2.GaussianBlur(M, (i % 2 * 2 + 1, i % 2 * 2 + 1), 0)
        real[:, int(rows / 2)-x - i:int(rows / 2) + x+i]=M
"""for i in range(int(rows/32),int(rows/32)*2):
    #L=real[:int(rows/2)-i,:]
    x= np.random.normal(loc=i*16, scale=2, size=None)
    x=int(x)
    M=real[int(rows / 2)-x:int(rows / 2)+x,int(rows / 2)-i:int(rows / 2)+i]
    #R=real[int(rows / 2)+i:,:]
    M=cv2.GaussianBlur(M,(i%2*2+1,i%2*2+1),0)
    real[int(rows / 2)-x:int(rows / 2)+x,int(rows / 2)-i:int(rows / 2)+i]=M"""

cv2.imshow('1',real)
cv2.waitKey(0)
for i in range(2,6):
    real=cv2.blur(real, (i,i))
    real=cv2.GaussianBlur(real, (i % 2 * 2 + 1, i % 2 * 2 + 1), 0)
cv2.imshow('1',real)
cv2.imwrite('direct_filter.jpg',real)
cv2.waitKey(0)
"""real = np.hstack((A[:, :int(cols / 2)], B[:, int(cols / 2):]))
for i in range(1,int(cols/32)):
    L=real[:,:int(cols/2)-i]
    M=real[:,int(cols / 2)-i:int(cols / 2)+i]
    R=real[:,int(cols / 2)+i:]
    M=cv2.cv2.blur(M, (i,i))
    real=np.hstack((L,M))
    real=np.hstack((real,R))
cv2.imshow('1',real)
cv2.waitKey(0)"""