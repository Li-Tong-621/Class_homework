import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
import sklearn.cluster  as cluster
from sklearn.metrics import fowlkes_mallows_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
iris = load_iris()
x = iris.data
y=iris.target
#print(y)
n=len(x)
#Kernel K-means
print("Kernel K-means:")
"""n_clusters:class的个数；
max_inter:每一个初始化值下，最大的iteration次数；
n_init:尝试用n_init个初始化值进行拟合;
tol:within-cluster sum of square to declare convergence;
init=‘k-means++’：可使初始化的centroids相互远离；"""
x_new=[]
for i in range(n):
    x_new.append(x[i][0]*x[i][1]*x[i][2]*x[i][3])
#print(x,x_new)
x_new=np.array(x_new)
x_new=x_new.reshape(-1,1)
max=0
mi=1
score=0
#根据评价指标选择最好的结果
for i in range(2,7):
    y_KMeans=KMeans(n_clusters=i,init='k-means++').fit_predict(x_new)
    score = fowlkes_mallows_score(y,y_KMeans)
    """score=sum(y_Gaussian == y)/n"""
    print("n_clusters={:}，FMI分数为{:}" .format(i,score))
    if score>max:
        mi=i
        max=score
print(mi,max)

#绘图
y_KMeans=KMeans(n_clusters=mi,init='k-means++').fit_predict(x_new)
plt.scatter(x_new, y,c=y_KMeans)
plt.show()



#利用高斯混合模型实现使用EM聚类方法对数据进行聚类
print("利用高斯混合模型实现使用EM聚类方法对数据进行聚类：")
"""n_components ：高斯模型的个数，即聚类的目标个数
covariance_type : 通过EM算法估算参数时使用的协方差类型，默认是"full"
full：每个模型使用自己的一般协方差矩阵
tied：所用模型共享一个一般协方差矩阵
diag：每个模型使用自己的对角线协方差矩阵
spherical：每个模型使用自己的单一方差"""
max=0
mi=1
score=0
#根据评价指标选择最好的结果
for i in range(2,7):
    y_Gaussian=GaussianMixture(n_components=i).fit_predict(x)
    score = fowlkes_mallows_score(y,y_Gaussian)
    """score=sum(y_Gaussian == y)/n"""
    print("n_components={:}，FMI分数为{:}" .format(i,score))
    if score>max:
        mi=i
        max=score
print(mi,max)

#绘图
y_Gaussian=GaussianMixture(n_components=mi).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_Gaussian)
plt.show()


#谱聚类
print("谱聚类：")
Spectral=cluster.SpectralClustering
"""n_clusters：聚类的个数。（官方的解释：投影子空间的维度）
affinity：核函数，默认是’rbf’，可选：“nearest_neighbors”，“precomputed”,
"rbf"或sklearn.metrics.pairwise_kernels支持的其中一个内核之一。
gamma :affinity指定的核函数的内核系数，默认1.0"""
max=0
mi=1
#根据评价指标选择最好的结果
for i in range(2,7):
    Spectral=cluster.SpectralClustering(n_clusters=i).fit(x)
    score=fowlkes_mallows_score(y,Spectral.labels_)
    #print(y)
    #print(Spectral.labels_)
    print("聚类%d簇的FMI分数为：%f" %(i,score))
    if score>max:
        mi=i
        max=score
print(mi,max)
#绘图
y_SpectralClustering = cluster.SpectralClustering(n_clusters=mi).fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=y_SpectralClustering)
plt.show()
