from sklearn.linear_model import LinearRegression  #线性回归模型
from sklearn.model_selection import train_test_split #切分训练集和测试集
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error #均方误差
from sklearn.metrics import mean_absolute_error #平方绝对误差
#from sklearn.metrics import r2_score
import pandas as pd
#使用pandas 读入数据并输出读入的数据
Boston = pd.read_csv('D:/文件/A大二课件下/机器学习/线性回归实验/boston.csv')  #,usecols=[0,5,12]
print('使用pandas 读入数据并输出读入的数据:')
print(Boston)
"""
x_data=[]
for i in ['crim','rm','lstat']:
    j= Boston[i]
    x_data.append(j)
print(x_data)
x_data = pd.read_csv('D:/文件/A大二课件下/机器学习/线性回归实验/boston.csv',usecols=[0,5,12])"""
#使用'crim', 'rm', 'lstat'作为特征值，'medv'为目标值，输出特征值的描述性统计
x_data=Boston[['crim', 'rm', 'lstat']]
y_data = Boston['medv']
print("使用'crim', 'rm', 'lstat'作为特征值，'medv'为目标值，输出特征值的描述性统计:")
print(x_data)
print(y_data)
#区分训练集和测试集	split_num = int(len(features)*0.7)
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size = 0.3,random_state = 0)
# 数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
x_train=min_max_scaler.fit_transform(x_train)
x_test=min_max_scaler.fit_transform(x_test)
y_train=min_max_scaler.fit_transform(y_train.values.reshape(-1,1))
y_test=min_max_scaler.fit_transform(y_test.values.reshape(-1,1))#reshape(-1,1)指将它转化为1列，行自动确定
#利用sklearn的LinearRegression()函数进行线性回归，输出模型的回归方程系数及方程的截距
lr=LinearRegression()
#使用训练数据进行参数估计
lr.fit(x_train,y_train)
print("系数矩阵")
print(lr.coef_)#系数矩阵
print("截距")
print(lr.intercept_)#截距
#回归预测
y_predict=lr.predict(x_test)
print("输出预测值：")
print(y_predict)
#score = r2_score(y_test, y_predict)
#print(score)
#求取预测值和真实值的mae和mse，调用函数
print("求取预测值和真实值的mae和mse:")
mse_test1 = mean_squared_error(y_test,y_predict)
mae_test1 = mean_absolute_error(y_test,y_predict)
print('Mse:{},Mae:{}'.format(mse_test1,mae_test1))
