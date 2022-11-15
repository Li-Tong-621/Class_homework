import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
class Neuro_net(torch.nn.Module):
    """搭建神经网络"""
    def __init__(self, n_feature, n_hidden_layer, n_output):
        super(Neuro_net, self).__init__()   # 继承__init__功能
        self.hidden_layer = torch.nn.Linear(n_feature, n_hidden_layer)
        self.output_layer = torch.nn.Linear(n_hidden_layer, n_output)

    def forward(self, x_data):
        hidden_layer_x = torch.relu(self.hidden_layer(x_data))
        pridect_y = self.output_layer(hidden_layer_x)
        return pridect_y

# 准备数据
x_data = pd.read_csv('x_train.csv')
x_data=torch.from_numpy(np.array(x_data)).float()
#x_data=np.array(x_data)
y_data = pd.read_csv('y_train.csv')
#y_data=np.array(y_data)
y_data=torch.from_numpy(np.array(y_data)).float()

# 通过matplotlib可视化生成的数据
# plt.scatter(x_data.numpy(), y_data.numpy())
# plt.show()

num_feature = 14
num_hidden_layer = 7
num_output = 1
epoch = 1000
# 实例化神经网络
net = Neuro_net(num_feature, num_hidden_layer, num_output)
# print(net)

# optimizer 优化
optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
# loss funaction
loss_funaction = torch.nn.MSELoss()

plt.ion()
# train
for step in range(epoch):
    pridect_y = net(x_data)  # 喂入训练数据 得到预测的y值
    loss = loss_funaction(pridect_y, y_data)  # 计算损失

    optimizer.zero_grad()    # 为下一次训练清除上一步残余更新参数
    loss.backward()          # 误差反向传播，计算梯度
    optimizer.step()         # 将参数更新值施加到 net 的 parameters 上

    if step % 5 == 0:
        print("已训练{}步 | loss：{}。".format(step, loss))
"""        plt.cla()
        plt.scatter(x_data.numpy(), y_data.numpy())
        plt.plot(x_data.numpy(), pridect_y.data.numpy(), color="green", marker="o", linewidth=6, label="predict_line")
        plt.text(0.5, 0, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 13, 'color': 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()"""
"""print(net.parameters())
params = list(net.named_parameters())
print(params.__len__())
print(params[0])
print(params[1])"""
y=net( torch.from_numpy(np.array([0,0,0,0,0,0,0,1,1,0,0,0,0,0])).float() )
print(y)
"""print(net)"""
torch.save(net,'model6_2.pth')
net = torch.load('model6_2.pth')

"""print(net.parameters())
params = list(net.named_parameters())
print(params.__len__())
print(params[0])
print(params[1])
"""
y=net( torch.from_numpy(np.array([0,0,0,0,0,0,0,1,1,0,0,0,0,0])).float() )
print(y)