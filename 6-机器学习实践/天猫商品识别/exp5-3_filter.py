#测验__________________________________________________________________
#______________________________________________________________________
import json
import pandas as pd
import mmcv
import argparse
def parse_args():
    parser=argparse.ArgumentParser(description='input')
    parser.add_argument('--now_json_I',default='./expresult/i_fs_cascade.json')
    parser.add_argument('--now_json_V',default='./expresult/v_fs_cascade.json')
    parser.add_argument('--will_json_I',default='./expresult/i_fs_cascade_filter.json')
    parser.add_argument('--will_json_V',default='./expresult/v_fs_cascade_filter.json')

    args=parser.parse_args()

    return args

args=parse_args()
now_json_I=args.now_json_I
now_json_V=args.now_json_V
will_json_I=args.will_json_I
will_json_V=args.will_json_V
#图片
#ann1=mmcv.load(now_json_I)
# res1={}
# for item in ann1['annotations']:
#     img_id=item['image_id']
#     if img_id not in res1:
#         res1[img_id]=[]
#     res1[img_id].append(item)
# ann_res1=[]
# id=0
# flag=False
# for key,items in res1.items():
#     for item in items:
#         if item['score']>0.8:
#             flag=True
#             break
#     if flag:
#         ann_res1+=items
#     flag=False

# ann1['annotations']=ann_res1
# mmcv.dump(ann1,will_json_I)


# 视频:
ann2=mmcv.load(now_json_V)
res2={}
for item in ann2['annotations']:
    img_id=item['image_id']
    if img_id not in res2:
        res2[img_id]=[]
    res2[img_id].append(item)
ann_res2=[]
id=0
flag=False
for key,items in res2.items():
    for item in items:
        if item['score']>0.8:
            flag=True
            break
    if flag:
        ann_res2+=items
    flag=False

ann2['annotations']=ann_res2
mmcv.dump(ann2,will_json_V)
print('5-3!')