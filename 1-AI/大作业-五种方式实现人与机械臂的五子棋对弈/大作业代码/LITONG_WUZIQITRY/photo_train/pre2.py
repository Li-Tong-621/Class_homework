import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
import cv2

X = []
Y = []



for i in range(1,390):
    image=cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//0//0 ('+str(i)+').jpg')
    hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
    X.append(((hist / 255).flatten()))
    Y.append(0)
for i in range(1,32):
    image=cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//1//1 ('+str(i)+').jpg')
    hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
    X.append(((hist / 255).flatten()))
    Y.append(1)
for i in range(1,39):
    image=cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//2//2 ('+str(i)+').jpg')
    hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
    X.append(((hist / 255).flatten()))
    Y.append(2)
X = np.array(X)
Y = np.array(Y)
#切分训练集和测试集
#print(X.shape,Y.shape)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
#随机率为100%选取其中的30%作为测试集
clf0 = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)

image=cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//1//1 (1).jpg')
hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
#hist=np.array(((hist / 255).flatten()))
hist=[(hist / 255).flatten()]
hist=np.array(hist)
p= clf0.predict(hist)
print(int(p))
#print(confusion_matrix(y_test, predictions0))
#print (classification_report(y_test, predictions0))

