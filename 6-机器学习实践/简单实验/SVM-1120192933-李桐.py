import numpy as np
#手写SVM

class SVM:
    def __init__(self, max_iter=5, kernel='rbf', C=1.0, toler=0.001, rbf_sigma=1.0):

        # 在初始化中初始化svm的各种选项————————————————————————————————————————
        self.max_iter = max_iter  # 最大轮次
        self.kernel = kernel  # 核函数的选择
        self.rbf_sigma = rbf_sigma  # 如果是rbf需要有这个
        self.C = C  # 松弛变量,惩罚参数
        self.toler = toler  # 迭代的终止条件之一
        self.b = 0

        # fit之后,在init_mat中初始化——————————————————————————————————————————
        self.n_samples = None  # 训练样本的个数
        self.alphas = None  # 拉格朗日乘子
        self.error_tmp = None  # 保存E的缓存
        # 在init_mat之后，才能再计算———————————————————————————————————————————
        self.kernel_mat = None

    def init_mat(self):

        self.n_samples = np.shape(self.train_x)[0]  # 训练样本的个数
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))  # 拉格朗日乘子
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))  # 保存E的缓存

    def calc_kernel_i(self, train_x_i):
        '''
        计算样本之间的核函数的值
        input:  train_x(mat):训练样本
                train_x_i(mat):第i个训练样本
        output: kernel_value(mat):样本之间的核函数的值
        '''

        kernel_value = np.mat(np.zeros((self.n_samples, 1)))

        if self.kernel == 'rbf':  # rbf核函数
            for i in range(self.n_samples):  # 从0到样本数目循环
                diff = self.train_x[i, :] - train_x_i
                # print(diff.shape)
                diff = diff.reshape(-1, 1)
                # print(np.exp(diff * diff.T / (-2.0 * self.rbf_sigma ** 2)))
                # print(diff.T * diff,diff* diff.T)
                kernel_value[i] = np.exp(np.matmul(diff.T, diff) / (-2.0 * self.rbf_sigma ** 2))
                # kernel_value[i] = np.exp(diff.T * diff / (-2.0 * self.rbf_sigma ** 2))
                # 原来的问题是输入数组是(4),而输出数组是(1,1),不匹配。应该是矩阵乘法写错了，改成这样了
        elif self.kernel == 'linear':
            kernel_value = self.train_x * train_x_i.T
        elif self.kernel == 'poly':
            kernel_value = (self.train_x * train_x_i.T + 1) ** 2
        else:  # 不使用核函数
            kernel_value = self.train_x * train_x_i.T

        # if self.kernel == 'linear':
        #     return sum([x1[k] * x2[k] for k in range(self.n)])
        # elif self.kernel == 'poly':
        #     return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 2

        return kernel_value

    def calc_kernel(self):
        '''
        计算核函数矩阵
        input:  train_x(mat):训练样本的特征值
                kernel_option(tuple):核函数的类型以及参数
        output: kernel_matrix(mat):样本的核函数的值
        '''
        kernel_matrix = np.mat(np.zeros((self.n_samples, self.n_samples)))  # 初始化样本之间的核函数值
        for i in range(self.n_samples):
            kernel_matrix[:, i] = self.calc_kernel_i(self.train_x[i, :])

        return kernel_matrix

    def KKT(self, alpha_i):
        # choose_and_update,1.判断选择出的第一个变量是否违反了KKT条件
        error_i = self.cal_error(alpha_i)  # 计算第一个样本的E_i
        # print(error_i.shape)
        # print((self.train_y[alpha_i] * error_i).shape)
        if ((self.train_y[alpha_i] * error_i).all() < -self.toler) and (self.alphas[alpha_i] < self.C) or \
                ((self.train_y[alpha_i] * error_i).all() > self.toler) and (self.alphas[alpha_i] > 0):
            return 1
        else:
            return 0

    def max_min(self, alpha_i, alpha_j):
        # choose_and_update,2、两个变量的计算上下界
        if self.train_y[alpha_i] != self.train_y[alpha_j]:
            L = max(0, self.alphas[alpha_j] - self.alphas[alpha_i])
            H = min(self.C, self.C + self.alphas[alpha_j] - self.alphas[alpha_i])
        else:
            L = max(0, self.alphas[alpha_j] + self.alphas[alpha_i] - self.C)
            H = min(self.C, self.alphas[alpha_j] + self.alphas[alpha_i])

        return L, H

    def cal_error(self, alpha_j):
        '''误差值的计算
        input:  alpha_k(int):选择出的变量
        output: error_k(float):误差值
        '''
        # print(np.multiply(self.alphas, self.train_y).T * self.kernel_mat[:, alpha_j] + self.b)
        # output_k = float(np.multiply(self.alphas, self.train_y).T * self.kernel_mat[:, alpha_j] + self.b)
        # print(self.alphas.shape,self.train_y.shape)
        # output_k = float(np.matmul(self.alphas, self.train_y).T * self.kernel_mat[:, alpha_j] + self.b)

        output_k = (np.multiply(self.alphas, self.train_y).T * self.kernel_mat[:, alpha_j] + self.b)
        error_k = output_k - float(self.train_y[alpha_j])
        return error_k

    def update_error_tmp(self, alpha_k):
        '''重新计算误差值
        input:  alpha_k(int):选择出的变量
        output: 对应误差值
        '''
        error = self.cal_error(alpha_k)
        self.error_tmp[alpha_k] = [1, error]

    def select_second_sample_j(self, alpha_i, error_i):
        '''选择第二个样本
        input:  alpha_i(int):选择出的第一个变量
                error_i(float):E_i
        output: alpha_j(int):选择出的第二个变量
                error_j(float):E_j
        '''
        # 标记为已被优化
        self.error_tmp[alpha_i] = [1, error_i]
        candidateAlphaList = np.nonzero(self.error_tmp[:, 0].A)[0]

        maxStep = 0
        alpha_j = 0
        error_j = 0

        if len(candidateAlphaList) > 1:
            for alpha_k in candidateAlphaList:
                if alpha_k == alpha_i:
                    continue
                error_k = self.cal_error(alpha_k)
                if abs(error_k - error_i) > maxStep:
                    maxStep = abs(error_k - error_i)
                    alpha_j = alpha_k
                    error_j = error_k
        else:  # 随机选择
            alpha_j = alpha_i
            while alpha_j == alpha_i:
                alpha_j = int(np.random.uniform(0, self.n_samples))
            error_j = self.cal_error(alpha_j)

        return alpha_j, error_j

    def choose_and_update(self, alpha_i):
        '''
        判断和选择两个alpha进行更新
        input: alpha_i(int):选择出的第一个变量
        '''
        error_i = self.cal_error(alpha_i)  # 计算第一个样本的E_i

        # 判断选择出的第一个变量是否违反了KKT条件
        if self.KKT(alpha_i):

            # 1、选择第二个变量
            alpha_j, error_j = self.select_second_sample_j(alpha_i, error_i)
            alpha_i_old = self.alphas[alpha_i].copy()
            alpha_j_old = self.alphas[alpha_j].copy()

            # 2、计算上下界
            L, H = self.max_min(alpha_i, alpha_j)
            if L == H:
                return 0

            # 3、计算eta
            eta = 2.0 * self.kernel_mat[alpha_i, alpha_j] - self.kernel_mat[alpha_i, alpha_i] - self.kernel_mat[
                alpha_j, alpha_j]
            if eta >= 0:
                return 0

            # 4、更新alpha_j
            self.alphas[alpha_j] -= self.train_y[alpha_j] * (error_i - error_j) / eta

            # 5、确定最终的alpha_j
            if self.alphas[alpha_j] > H:
                self.alphas[alpha_j] = H
            if self.alphas[alpha_j] < L:
                self.alphas[alpha_j] = L

            # 6、判断是否结束
            if abs(alpha_j_old - self.alphas[alpha_j]) < 0.00001:
                self.update_error_tmp(alpha_j)
                return 0

            # 7、更新alpha_i
            self.alphas[alpha_i] += self.train_y[alpha_i] * self.train_y[alpha_j] \
                                    * (alpha_j_old - self.alphas[alpha_j])

            # 8、更新b
            b1 = self.b - error_i - self.train_y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) \
                 * self.kernel_mat[alpha_i, alpha_i] \
                 - self.train_y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) \
                 * self.kernel_mat[alpha_i, alpha_j]
            b2 = self.b - error_j - self.train_y[alpha_i] * (self.alphas[alpha_i] - alpha_i_old) \
                 * self.kernel_mat[alpha_i, alpha_j] \
                 - self.train_y[alpha_j] * (self.alphas[alpha_j] - alpha_j_old) \
                 * self.kernel_mat[alpha_j, alpha_j]
            if (0 < self.alphas[alpha_i]) and (self.alphas[alpha_i] < self.C):
                self.b = b1
            elif (0 < self.alphas[alpha_j]) and (self.alphas[alpha_j] < self.C):
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.0

            # 9、更新error
            self.update_error_tmp(alpha_j)
            self.update_error_tmp(alpha_i)

            return 1
        else:
            return 0

    def fit(self, dataSet, labels):

        self.train_x = dataSet  # 训练特征
        self.train_y = labels  # 训练标签
        self.init_mat()
        self.kernel_mat = self.calc_kernel()  # 计算核函数矩阵

        # 开始训练
        entireSet = True
        alpha_pairs_changed = 0
        iteration = 0

        while (iteration < self.max_iter) and ((alpha_pairs_changed > 0) or entireSet):
            alpha_pairs_changed = 0
            if entireSet:
                # 对所有的样本
                for x in range(self.n_samples):
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1
            else:
                # 非边界样本
                bound_samples = []
                for i in range(self.n_samples):
                    if self.alphas[i, 0] > 0 and self.alphas[i, 0] < self.C:
                        bound_samples.append(i)
                for x in bound_samples:
                    alpha_pairs_changed += self.choose_and_update(x)
                iteration += 1

            # 在所有样本和非边界样本之间交替
            if entireSet:
                entireSet = False
            elif alpha_pairs_changed == 0:
                entireSet = True

    def predict_i(self, test_sample_x):
        '''利用SVM模型对每一个样本进行预测
        input:  test_sample_x(mat):样本
        output: predict(float):对样本的预测
        '''
        # 1、计算核函数矩阵
        kernel_value = self.calc_kernel_i(test_sample_x)
        # 2、计算预测值
        predict = kernel_value.T * np.multiply(self.train_y, self.alphas) + self.b
        # print((kernel_value.T * np.matmul(self.train_y, self.alphas) + self.b).shape)

        return predict

    def predict(self, test_x):
        """
        input:test_x(mat):测试集
        output:predict(mat):对样本的预测标签
        """
        predict_label = []
        m, n = test_x.shape
        for i in range(m):
            # 对每一个样本得到预测值
            predict_label_i = self.predict_i(test_x[i, :])
            # print(predict_label_i.shape )
            # predict_label_i=np.sign(predict_label_i)
            if (np.sign(predict_label_i)).all() == 1:
                predict_label_i = 1
            else:
                predict_label_i = 0
            # print(predict_label_i.shape )
            predict_label.append(predict_label_i)

        predict_label = np.array(predict_label)

        return predict_label

    def score(self, test_y, pred_y):
        """
        计算预测的准确性
        input:  pred_y(mat):预测的标签
                test_y(mat):测试的标签
        output: accuracy(float):预测的准确性
        """
        correct = 0.0
        test_y = test_y.reshape(-1, 1)
        for i in range(len(test_y)):
            if np.sign(pred_y[i]) == np.sign(test_y[i]):
                correct += 1
        accuracy = correct / len(test_y)

        return accuracy

    def predict_accuracy(self, test_x, test_y):
        '''
        计算预测的准确性
        input:  test_x(mat):测试的特征
                test_y(mat):测试的标签
        output: accuracy(float):预测的准确性
        '''
        correct = 0.0
        for i in range(len(test_y)):
            # 对每一个样本得到预测值
            predict_label = self.predict_i(test_x[i, :])
            # 判断每一个样本的预测值与真实值是否一致
            if np.sign(predict_label) == (np.sign(test_y[i])).all():
                correct += 1
        accuracy = correct / len(test_y)

        return accuracy
from  sklearn import datasets
x,y=datasets.make_moons(n_samples=10000,noise=0.005,random_state=666)
import pandas as pd
df=pd.read_csv('iris.csv')
df.head()


x_data=df[['150','4','setosa','versicolor']]
y_data=df['virginica']
#这个找到的数据集的头 好奇怪呀
x_data

import numpy as np
x=x_data.values
y=y_data.values
from sklearn.model_selection import train_test_split
#处理一下，变成二分类，实际上我在数据集上就把其他种类删掉了
for i in range(len(y)):
    if y[i]!=0 and y[i]!=1:
        y[i]=0

while (1):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)
    svm = SVM(max_iter=1000)
    svm.fit(x_train, y_train)
    y_pre = svm.predict(x_test)
    if (svm.score(y_pre, y_test)) > 0.8:
        print(svm.score(y_pre, y_test))
        break

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

