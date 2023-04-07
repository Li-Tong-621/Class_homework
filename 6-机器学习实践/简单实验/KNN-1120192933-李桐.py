#手写knn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report


class Knn():
    # 默认k=5，设置和sklearn中的一样
    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self.x = x
        self.y = y

    def predict(self, x_test):
        labels = []
        # 这里可以看出，KNN的计算复杂度很高，一个样本就是O(m * n)
        for i in range(len(x_test)):

            # 初始化一个y标签的统计字典
            dict_y = {}
            # 计算第i个测试数据到所有训练样本的欧氏距离
            diff = self.x - x_test[i]
            distances = np.sqrt(np.square(diff).sum(axis=1))

            # 对距离排名，取最小的k个样本对应的y标签
            rank = np.argsort(distances)
            rank_k = rank[:self.k]
            y_labels = self.y[rank_k]

            # 生成类别字典，key为类别，value为样本个数
            for j in y_labels:
                if j not in dict_y:
                    dict_y.setdefault(j, 1)
                else:
                    dict_y[j] += 1

            # 取得y_labels里面，value值最大对应的类别标签即为测试样本的预测标签

            # label = sorted(dict_y.items(),key = lambda x:x[1],reverse=True)[0][0]
            # 下面这种实现方式更加优雅
            label = max(dict_y, key=dict_y.get)

            labels.append(label)

        return labels

#生成数据
x,y = make_classification(n_features=2,n_redundant=0,random_state=2022)
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()
#预测
knn = Knn()
knn.fit(x,y)
labels = knn.predict(x)

#查看分类报告
print(classification_report(y,labels))



#画分类边缘
x_min,x_max = x[:,0].min() - 1,x[:,0].max() + 1
y_min,y_max = x[:,1].min() - 1,x[:,1].max() + 1

xx = np.arange(x_min,x_max,0.02)
yy = np.arange(y_min,y_max,0.02)

xx,yy = np.meshgrid(xx,yy)

x_1 = np.c_[xx.ravel(),yy.ravel()]
y_1 = knn.predict(x_1)

#list没有reshape方法，转为np.array的格式
plt.contourf(xx,yy,np.array(y_1).reshape(xx.shape),cmap='GnBu')
plt.scatter(x[:,0],x[:,1],c=y)
plt.show()


#KNN实验

import pandas as pd
import numpy as np
#1.数据导入/
lilac_data=pd.read_csv('D:/Pycode_2/机器学习实践/course-9-syringa.csv')
lilac_data.head()


#2.绘制特征子图，将特征两两组合，共有六种组合
from matplotlib import pyplot as plt
"""绘制丁香花特征子图
"""
fig, axes = plt.subplots(2, 3, figsize=(20, 10))  # 构建生成 2*3 的画布，2 行 3 列
fig.subplots_adjust(hspace=0.3, wspace=0.2)  # 定义每个画布内的行间隔和高间隔
axes[0, 0].set_xlabel("sepal_length")  # 定义 x 轴坐标值
axes[0, 0].set_ylabel("sepal_width")  # 定义 y 轴坐标值
axes[0, 0].scatter(lilac_data.sepal_length[:50],
                   lilac_data.sepal_width[:50], c="b")
axes[0, 0].scatter(lilac_data.sepal_length[50:100],
                   lilac_data.sepal_width[50:100], c="g")
axes[0, 0].scatter(lilac_data.sepal_length[100:],
                   lilac_data.sepal_width[100:], c="r")
axes[0, 0].legend(["daphne", "syringa", "willow"], loc=2)  # 定义示例

axes[0, 1].set_xlabel("petal_length")
axes[0, 1].set_ylabel("petal_width")
axes[0, 1].scatter(lilac_data.petal_length[:50],
                   lilac_data.petal_width[:50], c="b")
axes[0, 1].scatter(lilac_data.petal_length[50:100],
                   lilac_data.petal_width[50:100], c="g")
axes[0, 1].scatter(lilac_data.petal_length[100:],
                   lilac_data.petal_width[100:], c="r")

axes[0, 2].set_xlabel("sepal_length")
axes[0, 2].set_ylabel("petal_length")
axes[0, 2].scatter(lilac_data.sepal_length[:50],
                   lilac_data.petal_length[:50], c="b")
axes[0, 2].scatter(lilac_data.sepal_length[50:100],
                   lilac_data.petal_length[50:100], c="g")
axes[0, 2].scatter(lilac_data.sepal_length[100:],
                   lilac_data.petal_length[100:], c="r")

axes[1, 0].set_xlabel("sepal_width")
axes[1, 0].set_ylabel("petal_width")
axes[1, 0].scatter(lilac_data.sepal_width[:50],
                   lilac_data.petal_width[:50], c="b")
axes[1, 0].scatter(lilac_data.sepal_width[50:100],
                   lilac_data.petal_width[50:100], c="g")
axes[1, 0].scatter(lilac_data.sepal_width[100:],
                   lilac_data.petal_width[100:], c="r")

axes[1, 1].set_xlabel("sepal_length")
axes[1, 1].set_ylabel("petal_width")
axes[1, 1].scatter(lilac_data.sepal_length[:50],
                   lilac_data.petal_width[:50], c="b")
axes[1, 1].scatter(lilac_data.sepal_length[50:100],
                   lilac_data.petal_width[50:100], c="g")
axes[1, 1].scatter(lilac_data.sepal_length[100:],
                   lilac_data.petal_width[100:], c="r")

axes[1, 2].set_xlabel("sepal_width")
axes[1, 2].set_ylabel("petal_length")
axes[1, 2].scatter(lilac_data.sepal_width[:50],
                   lilac_data.petal_length[:50], c="b")
axes[1, 2].scatter(lilac_data.sepal_width[50:100],
                   lilac_data.petal_length[50:100], c="g")
axes[1, 2].scatter(lilac_data.sepal_width[100:],
                   lilac_data.petal_length[100:], c="r")


#3.数据集划分
from sklearn.model_selection import  train_test_split
feature_data=lilac_data.iloc[:,:-1]
label_data=lilac_data['labels']

X_train,X_test,y_train,y_test =train_test_split(feature_data,label_data,test_size=0.3,random_state=2)


#4.构建knn
from sklearn.neighbors import KNeighborsClassifier

def sklearn_classify(train_data,label_data,test_data,k_num):
    knn=KNeighborsClassifier(n_neighbors=k_num)
    knn.fit(train_data,label_data)
    predict_label=knn.predict(test_data)

    return predict_label

#5.测试
y_predict =sklearn_classify(X_train,y_train,X_test,3)
y_predict


#6.准确率
def get_accracy(test_labels,pred_labels):
    correct=np.sum(test_labels==pred_labels)
    n=len(test_labels)
    accur=correct/n
    return accur


#7.改变k值，绘制准确率随k值变化曲线

normal_accracy=[]
k_value=range(2,11)
for k in k_value:
    y_predict=sklearn_classify(X_train,y_train,X_test,k)
    accracy=get_accracy(y_test,y_predict)
    normal_accracy.append(accracy)


plt.xlabel('k')
plt.ylabel('accuracy')
new_ticks=np.linspace(0.6,0.9,10)
plt.yticks(new_ticks)
plt.plot(k_value,normal_accracy,c='r')
plt.grid('True')


#8————————————调参了
#8.2看一看distance
normal_accracy=[]
for i in k_value:
    knn=KNeighborsClassifier(n_neighbors=i,weights='distance')
    knn.fit(X_train,y_train)
    predict_label=knn.predict(X_test)
    accracy=get_accracy(y_test,y_predict)
    normal_accracy.append(accracy)

plt.xlabel('k')
plt.ylabel('accuracy')
new_ticks=np.linspace(0.6,0.9,10)
plt.yticks(new_ticks)
plt.plot(k_value,normal_accracy,c='r')
plt.grid('True')

# 8.3尝试寻找最好的p
best_p = -1
best_score = 0.0
best_k = -1
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, weights="distance", p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_p = p

print("best_k = " + str(best_k))
print("best_score = " + str(best_score))
print("best_p = " + str(best_p))

# 8.3尝试寻找最好的p
best_p = -1
best_score = 0.0
best_k = -1
for k in range(1, 11):
    for p in range(1, 6):
        knn_clf = KNeighborsClassifier(n_neighbors=k, p=p)
        knn_clf.fit(X_train, y_train)
        score = knn_clf.score(X_test, y_test)
        if score > best_score:
            best_k = k
            best_score = score
            best_p = p

print("best_k = " + str(best_k))
print("best_score = " + str(best_score))
print("best_p = " + str(best_p))
