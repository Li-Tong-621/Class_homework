import numpy as np
import sklearn.svm as svm  #导入svm函数
import matplotlib.pyplot as mp
from sklearn.model_selection import train_test_split #切分训练集和测试集
from sklearn.datasets import load_iris   #导入鸢尾花数据
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
iris = load_iris()
#s = iris.data[:,:2]  # 因为四维特征无法用散点图展示，所以只取前两维特征
x = iris.data
y = iris.target
# 可以看到样本大概分为三类
#print(s[:5])
print(x)
print(y)
# 基于svm 实现分类
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.3,random_state = 0)
model = svm.SVC(C=1,kernel='rbf',gamma=0.1)
model.fit(x,y)

num_test=len(y_test)

xl=[]
yl=[]
zl=[]
for i in range(10):
   for j in range(200):
      c=0.01*j+0.0001
      xl.append(i)
      clf_poly=svm.SVC(C=c,decision_function_shape="ovo", kernel="poly",degree=i)
      clf_poly.fit(x_train, y_train)
      y_predict_poly=clf_poly.predict(x_test)
      print("poly,C={:},degree={:}".format(c,i))
      print(y_predict_poly)
      acc_poly = sum(y_predict_poly == y_test) / num_test
      yl.append(c)
      zl.append(acc_poly)

   #l.append(c)
      print(acc_poly)
   # 画出图像"""'
"""plt.scatter(xl, yl, color='red',  linewidth=2)
plt.plot(xl, yl, color='green', linewidth=2)
plt.legend(loc='lower right')  # 设置标签的位置 可以尝试改为upper left
plt.show()"""
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(xl, yl, zl, c='r')  # 绘制数据点
ax.set_zlabel('rate of sucess')  # 坐标轴
ax.set_ylabel('c')
ax.set_xlabel('i')
plt.show()

w=0
xl=[]
yl=[]
zl=[]
for i in range(400):
   w=i*0.001+0.001
   for j in range(200):
      c=0.01*j+0.0001
      clf_linear = svm.SVC(C=c,decision_function_shape="ovo", kernel="linear", gamma=w)
      xl.append(w)
      clf_linear.fit(x_train, y_train)
      y_predict_linear = clf_linear.predict(x_test)
      print("linear,C={:},gamma={:}".format(c,w))
      print(y_predict_linear)
      acc_linear = sum(y_predict_linear == y_test) / num_test
      yl.append(c)
      zl.append(acc_linear)
      print(acc_linear)
   # 画出图像

ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(xl, yl, zl, c='r')  # 绘制数据点
ax.set_zlabel('rate of sucess')  # 坐标轴
ax.set_ylabel('c')
ax.set_xlabel('gamma')
plt.show()

xl=[]
yl=[]
zl=[]
for i in range(400):
   w=i*0.001+0.001
   for j in range(200):
      c=0.01*j+0.0001
      clf_rbf = svm.SVC(C=c,decision_function_shape="ovo", kernel="rbf",gamma=w)
      xl.append(w)
      clf_rbf.fit(x_train, y_train)
      y_predict_rbf = clf_rbf.predict(x_test)
      print("rbf,C={:},gamma={:}".format(c,w))
      print(y_predict_rbf)
      acc_rbf = sum(y_predict_rbf == y_test) / num_test
      zl.append(acc_rbf)
      yl.append(c)
      print(acc_rbf)
   # 画出图像
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(xl, yl, zl, c='r')  # 绘制数据点
ax.set_zlabel('rate of sucess')  # 坐标轴
ax.set_ylabel('c')
ax.set_xlabel('gamma')
plt.show()


xl=[]
yl=[]
zl=[]
for i in range(400):
   w=i*0.001+0.001
   for j in range(200):
      c=0.01*j+0.0001
      clf_sigmoid=svm.SVC(decision_function_shape="ovo", kernel="sigmoid",gamma=w)
      xl.append(w)
      clf_sigmoid.fit(x_train, y_train)
      y_predict_sigmoid = clf_sigmoid.predict(x_test)
      print("sigmoid,C={:},gamma={:}".format(c,w))
      print(y_predict_sigmoid)
      acc_sigmoid = sum(y_predict_sigmoid == y_test) / num_test
      zl.append(acc_sigmoid)
      yl.append(c)
      print(acc_sigmoid)
      # 画出图像
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(xl, yl, zl, c='r')  # 绘制数据点
ax.set_zlabel('rate of sucess')  # 坐标轴
ax.set_ylabel('c')
ax.set_xlabel('gamma')
plt.show()

"""clf_sigmoid=svm.SVC(decision_function_shape="ovo", kernel="sigmoid")
clf_sigmoid.fit(x_train, y_train)
y_predict_sigmoid = clf_sigmoid.predict(x_test)
print(y_predict_sigmoid)
acc_sigmoid = sum(y_predict_sigmoid == y_test) / num_test
print(acc_sigmoid)"""

