import torch.nn as nn
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import cv2
import json

                
# path='./data/i_train.json'
# with open(path, 'r') as f:
#     x = json.load(f)
# #print(x)
# for i in range(len(x['images'])):
#     print(x['images'][i]['id'])
#     #x=Image.open(x['images'][i]['file_name']).convert('RGB')
#     print(x['annotations'][i]['id'])
#     label=x['annotations'][i]['category_id']
class MYDataset(Dataset):
    def __init__(self,kind):
        """
        :参数 root_dir : 数据集所在文件夹路径
        :参数 img_dir  : 图片所在文件夹路径
        :参数 label_dir: 标签文件名
        :参数 enhancement: 数据增强的方法，包括None、以及方法1-7(int)
        """
        self.path='./data/i_train.json'
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

        img=Image.open(self.x['images'][i]['file_name']).convert('RGB')
        label = self.x['annotations'][i]['category_id']

        img = self.transform(img) #图片处理

        label=np.array(label)
        label=torch.from_numpy(label)#转tensor，变成onehot需要用tensor
        label=label.to(torch.int64)
        label = torch.nn.functional.one_hot(label, 23).float()
        return img, label

    def __len__(self):

        return 51

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        # 经过处理后的x要与x的维度相同(尺寸和深度)
        # 如果不相同，需要添加卷积+BN来变换为同一维度
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        pass

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


# 用于ResNet50,101和152的残差块，用的是1x1+3x3+1x1的卷积
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )
        pass

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        # 网络输入部分
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 中间卷积部分
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 平均池化和全连接层
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        pass

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 一个box_block中的图片大小只在第一次除以2
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = block.expansion * planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



import math
import torch
from torch.optim.optimizer import Optimizer

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
    with torch.no_grad():  # 该函数的意义需在实验报告中写明
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

dataset_train = MYDataset(kind='train')
train_loader = torch.utils.data.DataLoader(dataset_train,batch_size=10,shuffle=True,
                                           num_workers=0,pin_memory=True)
# 测试集
dataset_val =MYDataset(kind='eval')
val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=10, shuffle=True,
                                           num_workers=0,pin_memory=True)
epoches=20
import torchvision
model=torchvision.models.resnet50(pretrained=True)
model.fc=nn.Linear(model.fc.in_features,23)
model=model.to(device)

#x=torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
optimizer =Ranger(params=model.parameters())
criterion = nn.MSELoss().to(device)
test=[]
x_max=0
for epoch in range(20):
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
        loss = criterion(pre_labels, true_labels)
        #print(pre_labels.shape, true_labels.shape)
        # 后向传播
        loss.backward()
        # 更新模型
        optimizer.step()

        if index==0:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, index, loss.data.item()))

    # 手动释放内存
    del images, true_labels, pre_labels
    x=validation(data_loader=val_loader, model=model)
    print('acc:',x)
    test.append(x)
    if x>x_max:
        x_max=x
        # 保存训练模型
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, './data/4.pth')
    if epoch%30==0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.8
print(test)
print(acc)
