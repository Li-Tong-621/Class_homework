# 1.导入必要的库
import numpy as np
from matplotlib import pyplot as plt
def nms(bboxs, scores, threshold):
    x1 = bboxs[:, 0]
    y1 = bboxs[:, 1]
    x2 = bboxs[:, 2]
    y2 = bboxs[:, 3]
    areas = (y2 - y1) * (x2 - x1)  # 每个bbox的面积
 
    # order为排序后的得分对应的原数组索引值
    _, order = scores.sort(0, descending=True)
 
    keep = []  # 保存所有结果框的索引值。
    while order.numel() > 0:
        if order.numel() == 1:
            keep.append(order.item())
            break
        else:
            i = order[0].item()
            keep.append(i)
 
        # 计算最大得分的bboxs[i]与其余各框的IOU
        xx1 = x1[order[1:]].clamp(min=int(x1[i]))
        yy1 = y1[order[1:]].clamp(min=int(y1[i]))
        xx2 = x2[order[1:]].clamp(max=int(x2[i]))
        yy2 = y2[order[1:]].clamp(max=int(y2[i]))
        inter = (yy2 - yy1).clamp(min=0) * (xx2 - xx1).clamp(min=0)
        iou = inter / (areas[i] + areas[order[1:]] - inter)  
        # 如果bboxs长度为N，则iou长度为N-1
 
        # 保留iou小于阈值的剩余bboxs,.nonzero().squeeze()转化为数字索引，可验证
        idx = (iou <= threshold).nonzero().squeeze()
        if idx.numel() == 0:
            break
        order = order[idx + 1]  
        # idx+1表示对其每个值加一(保证和原来的order中索引一致)，并替换原来的order
 
    return keep
