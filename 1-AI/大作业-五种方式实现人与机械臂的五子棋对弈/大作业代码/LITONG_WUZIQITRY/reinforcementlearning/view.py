from environment import env
import numpy as np
import time
from DQN2 import DeepQNetwork
import tensorflow as tf
space=361
space_col=19

class view():
    def KK(board):
        gameStart=False
        status=False
        reward=0
        n_actions = 361    #定义动作的可能个数
        n_features = 361
        doneList=[]
        allphoto=[]
        wobservation=None
        wobservation_=None
        action1=None
        tf.reset_default_graph()
        RL = DeepQNetwork(n_actions, n_features )

        qipan=board.copy()
        observation=view.getdouble(qipan)
        action=RL.choose_action(qipan,observation)
        i=action//19
        j=action%19
        return(i,j)


    def tryPosition(Ob,ation,flag):
         qipan=np.copy(Ob)
         if flag=='White':
             qipan[0,ation]=1
         else:
             qipan[0,ation]=2
         return qipan


    def transfore(observation):
        # print(np.shape(shape)[1])
        s1=observation[0,:space]
        s2=observation[0,space:]
        s=np.hstack((s1,s2))
        return s

    #将棋盘1*361转化为1*722形式
    def getdouble(qipan):
        w_qipan=np.zeros([1,space])
        b_qipan=np.zeros([1,space])
        w_array=np.where(qipan==1)[1]
        b_array=np.where(qipan==2)[1]
        w_qipan[0,w_array]=1
        b_qipan[0,b_array]=1
        s=np.hstack((w_qipan,b_qipan))  #转化为1*722矩阵，前361是白字的状态，后361是黑子的状态
        return s

