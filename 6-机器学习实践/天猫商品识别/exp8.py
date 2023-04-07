
from ResNeStmaster.resnest.torch import resnest50
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
#print(1)
class MYDataset(Dataset):
    def __init__(self,kind):
        """
        :参数 root_dir : 数据集所在文件夹路径
        :参数 img_dir  : 图片所在文件夹路径
        :参数 label_dir: 标签文件名
        :参数 enhancement: 数据增强的方法,包括None、以及方法1-7(int)
        """
        if kind=='trian':
            self.path='./data3/i_train.json'
            self.rpath='./data3/I/'
        else:
            self.path='./data4/i_train.json'
            self.rpath='./data4/I/'
        #path = './data/i_train.json'
        with open(self.path, 'r') as f:
            self.x = json.load(f)
        
        #数据处理
        if kind=='train':
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])  # 归一化
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])])  # 归一化
    def __getitem__(self, i):

        img=Image.open(self.rpath+self.x['images'][i]['file_name']).convert('RGB')
        label = self.x['annotations'][i]['category_id']
        #print(label)
        img = self.transform(img) #图片处理

        label=np.array(label)
        
        label=torch.from_numpy(label)#转tensor，变成onehot需要用tensor
        label=label.to(torch.int64)
        label = torch.nn.functional.one_hot(label, 24).float()
        return img, label

    def __len__(self):

        return len(self.x['images'])

class MYDataset2(Dataset):
    def __init__(self,kind):
        """
        :参数 root_dir : 数据集所在文件夹路径
        :参数 img_dir  : 图片所在文件夹路径
        :参数 label_dir: 标签文件名
        :参数 enhancement: 数据增强的方法,包括None、以及方法1-7(int)
        """
        if kind=='trian':
            self.path='./data3/i_train.json'
            self.rpath='./data3/I/'
        else:
            self.path='./data4/i_train.json'
            self.rpath='./data4/I/'
        #path = './data/i_train.json'
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
        label = self.x['annotations'][i]['category_id']
        #print(label)
        img = self.transform(img) #图片处理

        label=np.array(label)
        
        label=torch.from_numpy(label)#转tensor，变成onehot需要用tensor
        label=label.to(torch.int64)
        label = torch.nn.functional.one_hot(label, 24).float()
        return img, label

    def __len__(self):

        return len(self.x['images'])



def centralized_gradient(x, use_gc=True, gc_conv_only=False):
    if use_gc:
        if gc_conv_only:
            if len(list(x.size())) > 3:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
        else:
            if len(list(x.size())) > 1:
                x.add_(-x.mean(dim=tuple(range(1, len(list(x.size())))), keepdim=True))
    return x


class Ranger(Optimizer):

    def __init__(self, params, lr=1e-3,                       # lr
                 alpha=0.5, k=6, N_sma_threshhold=5,           # Ranger options
                 betas=(.95, 0.999), eps=1e-5, weight_decay=1e-5,  # Adam options
                 use_gc=True, gc_conv_only=False, gc_loc=True
                 ):

        # parameter checks
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f'Invalid slow update rate: {alpha}')
        if not 1 <= k:
            raise ValueError(f'Invalid lookahead steps: {k}')
        if not lr > 0:
            raise ValueError(f'Invalid Learning Rate: {lr}')
        if not eps > 0:
            raise ValueError(f'Invalid eps: {eps}')

        # prep defaults and init torch.optim base
        defaults = dict(lr=lr, alpha=alpha, k=k, step_counter=0, betas=betas,
                        N_sma_threshhold=N_sma_threshhold, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

        # adjustable threshold
        self.N_sma_threshhold = N_sma_threshhold

        # look ahead params

        self.alpha = alpha
        self.k = k

        # radam buffer for state
        self.radam_buffer = [[None, None, None] for ind in range(10)]

        # gc on or off
        self.gc_loc = gc_loc
        self.use_gc = use_gc
        self.gc_conv_only = gc_conv_only
        # level of gradient centralization
        #self.gc_gradient_threshold = 3 if gc_conv_only else 1

        print(
            f"Ranger optimizer loaded. \nGradient Centralization usage = {self.use_gc}")
        if (self.use_gc and self.gc_conv_only == False):
            print(f"GC applied to both conv and fc layers")
        elif (self.use_gc and self.gc_conv_only == True):
            print(f"GC applied to conv layers only")

    def __setstate__(self, state):
        print("set state called")
        super(Ranger, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data.float()

                if grad.is_sparse:
                    raise RuntimeError(
                        'Ranger optimizer does not support sparse gradients')

                p_data_fp32 = p.data.float()

                state = self.state[p]  # get state dict for this param

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p_data_fp32)
                    state['exp_avg_sq'] = torch.zeros_like(p_data_fp32)

                    # look ahead weight storage now in state dict
                    state['slow_buffer'] = torch.empty_like(p.data)
                    state['slow_buffer'].copy_(p.data)

                else:
                    state['exp_avg'] = state['exp_avg'].type_as(p_data_fp32)
                    state['exp_avg_sq'] = state['exp_avg_sq'].type_as(
                        p_data_fp32)

                # begin computations
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                # GC operation for Conv layers and FC layers
                # if grad.dim() > self.gc_gradient_threshold:
                #    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))
                if self.gc_loc:
                    grad = centralized_gradient(grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)

                state['step'] += 1

                # compute variance mov avg
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # compute mean moving avg
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                buffered = self.radam_buffer[int(state['step'] % 10)]

                if state['step'] == buffered[0]:
                    N_sma, step_size = buffered[1], buffered[2]
                else:
                    buffered[0] = state['step']
                    beta2_t = beta2 ** state['step']
                    N_sma_max = 2 / (1 - beta2) - 1
                    N_sma = N_sma_max - 2 * \
                        state['step'] * beta2_t / (1 - beta2_t)
                    buffered[1] = N_sma
                    if N_sma > self.N_sma_threshhold:
                        step_size = math.sqrt((1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (
                            N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    else:
                        step_size = 1.0 / (1 - beta1 ** state['step'])
                    buffered[2] = step_size

                # apply lr
                if N_sma > self.N_sma_threshhold:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    G_grad = exp_avg / denom
                else:
                    G_grad = exp_avg

                if group['weight_decay'] != 0:
                    G_grad.add_(p_data_fp32, alpha=group['weight_decay'])
                # GC operation
                if self.gc_loc == False:
                    G_grad = centralized_gradient(G_grad, use_gc=self.use_gc, gc_conv_only=self.gc_conv_only)
                p_data_fp32.add_(G_grad, alpha=-step_size * group['lr'])
                p.data.copy_(p_data_fp32)

                # integrated look ahead...
                # we do it at the param level instead of group level
                if state['step'] % group['k'] == 0:
                    # get access to slow param tensor
                    slow_p = state['slow_buffer']
                    # (fast weights - slow weights) * alpha
                    slow_p.add_(p.data - slow_p, alpha=self.alpha)
                    # copy interpolated weights to RAdam param tensor
                    p.data.copy_(slow_p)
        return loss
class CrossEntropyLabelSmooth(nn.Module):
    """
    用于指定带标签平滑的交叉熵公式 ,用于指定带标签平滑的交叉熵公式
    """

    def __init__(self, num_classes, epsilon=0.2, use_gpu=True):
        """
        __init__()方法参数有num_classes与epsilon
        第一个参数指定分类数量
        第二参数即标签平滑公式中的epsilon  这里是对应的标签平滑的过程的。
        """
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes  # num_classes = 8
        self.epsilon = epsilon # epsilon = 0.1
        self.use_gpu = use_gpu # 是否是要来使用gpu的过程的。
        self.logsoftmax = nn.LogSoftmax(dim=1) # 把相应的Softmax在来通过log的形式的。

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes) 是来对应的真实的标签的。
        """
        log_probs = self.logsoftmax(inputs)  # torch.Size([4, 8])这里是得到批次为4，每一个属于这8个类别中那一个概率是最大的
        # scatter_是来沿着1，列方向上这个维度来进行索引的。zeros[4,8]的列数要与scatter这边的列数是相同的。
        # ongTensor中的index最大值应与zeros(4, 8)行数相一致
        #targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data, 1) # 这里是来输出相应的标签信息的。
        #torch.topk(pre_labels, 1)[1].squeeze(1)
        #targets = torch.zeros(log_probs.size()).scatter_(1, torch.topk(log_probs, 1)[1].squeeze(1).data, 1) # 这里是来输出相应的标签信息的。
        #print(targets)
        #if self.use_gpu: 
        #targets = targets.cuda()  #如果是存在gpu话，就是放在gpu上面来进行计算的过程的。
        # y = (1 - epsilon) * y + epsilon / K.
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes  # 这里是来对应的标签平滑化的过程的。
        #print(targets)
        loss = (- targets * log_probs).mean(0).sum()
        #print(1)
        return loss
class TripletLoss(nn.Module):
    """Triplet loss with hard positive/negative mining.

    """
    def __init__(self, margin=0.3):#三元组的阈值margin
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)#三元组损失函数
        #ap an margin y:倍率   Relu(ap - anxy + margin)这个relu就起到和0比较的作用

    def forward(self, inputs, targets):
        """
        Args:
            inputs: visualization_feature_map matrix with shape (batch_size, feat_dim)#32x2048
            targets: ground truth labels with shape (num_classes)#tensor([32])[1,1,1,1,2,3,2,,,,2]32个数，一个数代表ID的真实标签
        """
        n = inputs.size(0)#取出输入的batch
        # Compute pairwise distance, replace by the official when merged
        #计算距离矩阵，其实就是计算两个2048维之间的距离平方(a-b)**2=a^2+b^2-2ab
        #[1,2,3]*[1,2,3]=[1,4,9].sum()=14  点乘

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())#生成距离矩阵32x32，.t()表示转置
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability#clamp(min=1e-12)加这个防止矩阵中有0，对梯度下降不好
        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())#利用target标签的expand，并eq，获得mask的范围，由0，1组成，，红色1表示是同一个人，绿色0表示不是同一个人
        dist_ap, dist_an = [], []#用来存放ap，an
        for i in range(n):#i表示行
            # dist[i][mask[i]],,i=0时，取mask的第一行，取距离矩阵的第一行，然后得到tensor([1.0000e-06, 1.0000e-06, 1.0000e-06, 1.0000e-06])
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))#取某一行中，红色区域的最大值，mask前4个是1，与dist相乘
            #print(dist[i],mask[i]  )
            #print(dist[i][mask[i] == 0].min().unsqueeze(0))
            try:
                dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))#取某一行，绿色区域的最小值,加一个.unsqueeze(0)将其变成带有维度的tensor
            except:
                #print(dist_an)#tensor([1.9137], device='cuda:0', grad_fn=<UnsqueezeBackward0>)
                #print(torch.mean(torch.tensor(dist_an)).view(-1))
                dist_an.append(torch.mean(torch.tensor(dist_an)).view(-1).cuda())
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)
        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)#y是个权重，长度像dist-an
        loss = self.ranking_loss(dist_an, dist_ap, y) #ID损失：交叉商输入的是32xf f.shape=分类数,然后loss用于计算损失
                                                      #度量三元组：输入的是dist_an（从距离矩阵中，挑出一行（即一个ID）的最大距离），dist_ap
                                                     #ranking_loss输入 an ap margin y:倍率  loss： Relu(ap - anxy + margin)这个relu就起到和0比较的作用
        # from IPython import embed
        # embed()
        return loss

class MultiSimilarityLoss(nn.Module):
    def __init__(self, margin=0.7):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 0.5
        self.margin = margin

        self.scale_pos = 2.0
        self.scale_neg = 40.0

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        feats = nn.functional.normalize(feats, p=2, dim=1)

        # Shape: batchsize * batch size
        sim_mat = torch.matmul(feats, torch.t(feats))

        epsilon = 1e-5
        loss = list()

        mask = labels.expand(batch_size, batch_size).eq(
            labels.expand(batch_size, batch_size).t())
        for i in range(batch_size):
            pos_pair_ = sim_mat[i][mask[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = sim_mat[i][mask[i] == 0]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            #print(neg_pair_)
            try:
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            except:
                pos_pair = pos_pair_[pos_pair_ - self.margin < max(torch.tensor([0]))]

            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            pos_loss = 1.0 / self.scale_pos * torch.log(
                1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))
            loss.append(pos_loss + neg_loss)
            # pos_loss =


        if len(loss) == 0:
            return torch.zeros([], requires_grad=True, device=feats.device)

        loss = sum(loss) / batch_size
        return loss




acc=[]
def validation(data_loader,model,checkpoint=None):
    """
    # 验证函数，任务是在训练经过一定的轮数之后，对验证集中的数据进行预测并与真实结果进行比对，生成当前模型在验证集上的准确率
    :参数 data_loader:样本数据
    :参数 model:模型
    :参数 checkpoint:预训练模型,若有,则给出模型路径,在此模型基础上测试
    """

    model = model.to(device)
    # 加载预训练模型
    global start_epoch
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])

    # 自适应设备
    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))
    # 准备完毕，开始测试了
    correct = 0
    total = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():  
        for data in data_loader:

            images, true_labels = data
            images = images.to(device)
            true_labels = true_labels.to(device)

            # 前向传播
            pre_labels = model(images)
            total += pre_labels.shape[0]#总测试数

            #判断正误
            for i in range(pre_labels.shape[0]):
                #print(pre_labels[i],torch.argmax(pre_labels[i]) ,torch.argmax(true_labels[i]))
                #print(torch.argmax(pre_labels[i]) ,torch.argmax(true_labels[i]))
                if torch.argmax(pre_labels[i]) == torch.argmax(true_labels[i]):
                    correct += 1

    print("验证集数据总量：", total, "预测正确的数量：", correct)
    accuracy=correct/total
    print("当前模型在验证集上的准确率为：", accuracy)
    global acc
    acc.append(accuracy)
    return accuracy
# 设备参数
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# gpu数量
ngpu = torch.cuda.device_count()

bs=24
dataset_train = MYDataset(kind='train')

train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=bs,shuffle=True,
                                           num_workers=4,pin_memory=True,drop_last=True)
# dataset_train2 = MYDataset2(kind='train')
# train_loader2 = torch.utils.data.DataLoader(dataset_train2,batch_size=bs,shuffle=True,
#                                            num_workers=4,pin_memory=True,drop_last=True)
# 测试集
dataset_val =MYDataset(kind='eval')
#print(len(dataset_train),len(dataset_val))
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=bs, shuffle=True,
                                           num_workers=4,pin_memory=True,drop_last=True)
epoches=2
def train(path,model,optimizerkind=None,criterionkind=None,warmkind=None):
    acc=[]
    x_max=0
    init_lr=0.01
    learning_rate=0.01
    warm_up_epochs=int(epoches*0.2)
    if optimizerkind=='Adam':
        optimizer=torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    elif optimizerkind=='Ranger':
        optimizer =Ranger(params=model.parameters())

    if criterionkind=='cross_entropy':
        #criterion=nn.CrossEntropyLoss().to(device)
        criterion=CrossEntropyLabelSmooth(num_classes=23,epsilon=0)
    elif criterionkind=='CrossEntropyLabelSmooth':
        criterion=CrossEntropyLabelSmooth(num_classes=23)
    elif criterionkind=='MultiSimilarityLoss':
        criterion=MultiSimilarityLoss()
    elif criterionkind=='TripletLoss':
        criterion=TripletLoss()

    for epoch in range(50):
        model.train()  # 训练模式：允许使用批样本归一化
        # 循环外可以自行添加必要内容
        for index, data in enumerate(train_loader, 0):
            images, true_labels = data
            images=images.to(device)
            true_labels=true_labels.to(device)

            optimizer.zero_grad()
            # 前向传播
            pre_labels=model(images)
            # 计算损失
            #if criterionkind=='cross_entropy' or 'CrossEntropyLabelSmooth':
                #print(torch.argmax(true_labels))
                #pre_labels= torch.topk(pre_labels, 1)[1].squeeze(1)
                #true_labels=torch.topk(true_labels, 1)[1].squeeze(1)
                #pre_labels= torch.topk(pre_labels, 1)[1]
                #true_labels=torch.topk(true_labels, 1)[1]
                #pre_labels=pre_labels.long()
                #true_labels=true_labels.long()
                #print(pre_labels,true_labels)
                #pre_labels=pre_labels.float()
                #true_labels=true_labels.float()
           
            loss = criterion(pre_labels, true_labels)
            #print(pre_labels.shape, true_labels.shape)
            # 后向传播
            loss.backward()
            # 更新模型
            optimizer.step()

            if index==0:
                print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, index, loss.data.item()))

        # 手动释放内存
        #del images, true_labels, pre_labels
        x=validation(data_loader=val_loader, model=model)
        print('acc:',x)
        acc.append(x)
        if x>x_max:
            x_max=x
            # 保存训练模型
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, path+'.pth')
        if warm_up_epochs and epoch < warm_up_epochs:
            warmup_percent_done = epoch / warm_up_epochs
            warmup_learning_rate = init_lr * warmup_percent_done  #gradual warmup_lr
            learning_rate = warmup_learning_rate
        else:
            if warmkind=='cos':
                learning_rate = np.sin(learning_rate)  #预热学习率结束后,学习率呈sin衰减
            elif warmkind=='multistp':
                learning_rate = learning_rate**1.0001 #预热学习率结束后,学习率呈指数衰减(近似模拟指数衰减)
        for p in optimizer.param_groups:
                p['lr'] = learning_rate


        # if epoch%30==0:
        #     for p in optimizer.param_groups:
        #         p['lr'] *= 0.8
    #print(test)
    print(acc)
    f = open(path+'.txt', "w+")
    f.write(str(acc))
    f.close()


#x=torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)

#数据未增强的:
# model= resnest50(pretrained=True)
# for param in model.parameters():
#     param.requires_grad=False
#     model.fc=nn.Linear(model.fc.in_features,24)
# #model.fc=nn.Linear(model.fc.in_features,24)
# model=model.to(device)
# train(path='results2/4-unenhance',model=model,optimizerkind='Adam',criterionkind='cross_entropy',warmkind='multistp')


# for i in ['Adam','Ranger']:
#     model= resnest50(pretrained=True)
#     for param in model.parameters():
#         param.requires_grad=False
#         model.fc=nn.Linear(model.fc.in_features,24)
#     #model.fc=nn.Linear(model.fc.in_features,24)
#     model=model.to(device)
#     train(path='results2/1-optimizer-'+i,model=model,optimizerkind=i,criterionkind='cross_entropy',warmkind='multistp')
# #for i in ['cross_entropy','CrossEntropyLabelSmooth','MultiSimilarityLoss','TripletLoss']:
# for i in ['cross_entropy','CrossEntropyLabelSmooth']:
#     model= resnest50(pretrained=True)
#     for param in model.parameters():
#         param.requires_grad=False
#         model.fc=nn.Linear(model.fc.in_features,24)
#     model=model.to(device)
#     train(path='results2/2-criterion-'+i,model=model,optimizerkind='Ranger',criterionkind=i,warmkind='multistp')

#for i in ['MultiSimilarityLoss','TripletLoss']:
for i in ['TripletLoss']:
    model= resnest50(pretrained=True)
    for param in model.parameters():
        param.requires_grad=False
        model.fc=nn.Linear(model.fc.in_features,24)
    model=model.to(device)
    train(path='results2/2-criterion-'+i,model=model,optimizerkind='Ranger',criterionkind=i,warmkind='multistp')


# model= resnest50(pretrained=True)
# for param in model.parameters():
#     param.requires_grad=False
#     model.fc=nn.Linear(model.fc.in_features,24)
# model=model.to(device)
# train(path='results2/3-warm-cos',model=model,optimizerkind='Ranger',criterionkind='CrossEntropyLabelSmooth',warmkind='cos')

