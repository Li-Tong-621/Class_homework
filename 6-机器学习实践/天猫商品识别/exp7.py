
import argparse
import numpy as np
import faiss           
from ResNeStmaster.resnest.torch import resnest50,resnest50_m
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim.optimizer import Optimizer
import numpy as np
from PIL import Image
import torch.nn as nn
import torch
import torchvision
import os
import cv2
import json
import math
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu数量
ngpu = torch.cuda.device_count()
def parse_args():
    parser=argparse.ArgumentParser(description='input checkpont')
    parser.add_argument('--checkponit',default='./expresult/rec.pth')
    #python try.py --checkponit='./results2/1-optimizer-Adam.pth','./results2/1-optimizer-Adam.pth'#多个的话可以这样输入
    parser.add_argument('--I_json_path',default='./data3/i_train.json')
    parser.add_argument('--V_json_path',default='./expresult/v_fs_cascade_filter.json')
    parser.add_argument('--I_path',default='./data3/I/')
    parser.add_argument('--V_path',default='./data3/V/')
    parser.add_argument('--output_path',default='./expresult/topk.txt')

    args=parser.parse_args()

    return args


def Query(q,g,topK,nlist=24):
    #nlist=# 聚类质心有多少
    q=np.array(q)
    g=np.array(g)
    d=q.shape[1]#维度
    quantizer = faiss.IndexFlatL2(d)  # IndexFlatL2 & IndexFlatIP -> 欧式距离 & 内积搜索
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    assert not index.is_trained
    index.train(q)
    assert index.is_trained

    index.add(q)
    # METRIC_L2 & METRIC_INNER_PRODUCT -> 欧式距离 & 内积搜索


    k = 4                         # we want to see 4 nearest neighbors
    #[nq, k]  对query的每行找到k个相应的distance和index
    D, I = index.search(g, k)     # actual search (距离, ID)
    # print(I[:5])    # 输出前5行
    # print(D[-5:])   # 输出后5行
    return I[:topK]

class MYDataset(Dataset):
    def __init__(self,jsonpath,path):
        """
        :参数 root_dir : 数据集所在文件夹路径
        :参数 img_dir  : 图片所在文件夹路径
        :参数 label_dir: 标签文件名
        :参数 enhancement: 数据增强的方法,包括None、以及方法1-7(int)
        """
        self.path='./data3/i_train.json'
        self.path=jsonpath

        self.rpath='./data3/I/'
        self.rpath=path

        with open(self.path, 'r') as f:
            self.x = json.load(f)
        
        #数据处理
        self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])  # 归一化

    def __getitem__(self, i):
        
        img=Image.open(self.rpath+self.x['images'][i]['file_name']).convert('RGB')
        try:
            label = self.x['annotations'][i]['category_id']
        except:
            label=0
        #print(label)
        img = self.transform(img) #图片处理

        label=np.array(label)
        
        label=torch.from_numpy(label)#转tensor，变成onehot需要用tensor
        label=label.to(torch.int64)
        label = torch.nn.functional.one_hot(label, 24).float()
        return img, label

    def __len__(self):

        return len(self.x['images'])



def validation(data_loader,model,checkpoint=None):
    """
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    :参数 data_loader:样本数据
    :参数 model:模型
    :参数 checkpoint:预训练模型,若有,则给出模型路径,在此模型基础上测试
    """

    model = model.to(device)
    # 加载预训练模型
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint['model'])

    # 自适应设备
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))
    # 准备完毕，开始测试了
    model.eval()
    feature_list=[]
    with torch.no_grad():  
        for data in data_loader:

            images, true_labels = data
            images = images.to(device)
            true_labels = true_labels.to(device)

            # 前向传播
            model_feature = model(images)
            model_feature=model_feature.to('cpu')
            model_feature=model_feature.numpy()
            for i in model_feature:
                feature_list.append(i)



    return feature_list

args=parse_args()
model= resnest50_m(pretrained=True)
for param in model.parameters():
    param.requires_grad=False
    model.fc=nn.Linear(model.fc.in_features,24)
    
bs=200
dataset_train = MYDataset(jsonpath=args.I_json_path,path=args.I_path)
print(1)
dataset_val = MYDataset(jsonpath=args.V_json_path,path=args.V_path)

# print(len(dataset_train))
# print(len(dataset_val))

train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=bs,shuffle=False,
                                           num_workers=4,pin_memory=True,drop_last=False)
val_loader = torch.utils.data.DataLoader(dataset_val,batch_size=bs,shuffle=False,
                                           num_workers=4,pin_memory=True,drop_last=False)
                                        
check_list=args.checkponit.split(',')
ans_list=[]
for i in check_list:
    q=validation(data_loader=train_loader,model=model,checkpoint=i)
    print('训练数据准备好了')
    g=validation(data_loader=val_loader,model=model,checkpoint=i)
    print('准备！')
    ans=Query(q=q,g=g,topK=2)
    ans_list.append(ans.reshape(-1))
np.savetxt(args.output_path,ans_list,delimiter=',') #frame: 文件 array:存入文件的数组


