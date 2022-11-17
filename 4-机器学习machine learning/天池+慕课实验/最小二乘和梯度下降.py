import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

lg = LinearRegression()
# (x,y)的值 把x和y都reshape成列向量
x = np.array([55, 71, 68, 87, 101, 87, 75, 78, 93, 73]).reshape(-1, 1)
y = np.array([91, 101, 87, 109, 129, 98, 95, 101, 104, 93]).reshape(-1, 1)
# 训练模型
lg.fit(x, y)
# 返回预测的y值
y_predict = lg.predict(x)
# 输出权重

c=lg.intercept_[0]
d=lg.coef_[0][0]
print("最小二乘：w1={:},w0={:},y={:}*x+{:}".format(d,c,d,c))
# 画出图像
plt.scatter(x, y, color='red', label="sample data", linewidth=2)
plt.plot(x, y_predict, color='green', label="fitting line", linewidth=2)
plt.legend(loc='lower right')  # 设置标签的位置 可以尝试改为upper left
plt.show()

x = np.array([55, 71, 68, 87, 101, 87, 75, 78, 93, 73])
yy = np.array([91, 101, 87, 109, 129, 98, 95, 101, 104, 93])
#print('准确的训练数据 = ', yy)
# 2 构建初始模型
a = 0.7
b = 50
yy_pridict = a * x + b
#print('模型的生成数据 = ', yy_pridict)
loss = np.sum(np.square(yy_pridict - yy))/10
i=1
# 3 迭代更新  数据只有十个 所以迭代次数多一些
while loss >0.0001:
    a = a - 0.0002 * np.dot((yy_pridict - yy), x.transpose())/10
    b = b - 0.0002 * np.sum(yy_pridict - yy)/10
    yy_pridict = a * x + b
    temp=loss
    loss = np.sum(np.square(yy_pridict - yy))/10/2
    #print("%d：a=%f b=%f loss=%f " % (i + 1, a, b, loss))
    i=i+1
    if loss==temp:
        print("梯度下降第%d次迭代：a=%f b=%f loss=%f " % (i + 1, a, b, loss))
        break

print("最小二乘：w0={:.3},w1={:.3},y={:.3}+{:.3}*x".format(c,d,c,d))
print("梯度下降：w0={:.3},w1={:.3},y={:.3}+{:.3}*x".format(b,a,b,a))
yy_pridict = a * x + b
plt.scatter(x, yy, color='red', label="sample data", linewidth=2)
plt.plot(x, yy_pridict, color='green', label="fitting line", linewidth=2)
plt.legend(loc='lower right')  # 设置标签的位置 可以尝试改为upper left
plt.show()
