import argparse 
import os
import glob
import collections
import json
from tqdm import tqdm
import cv2
import numpy as np
#获得dataset名称
#name=args.dataset
def parse_args():
    parser=argparse.ArgumentParser(description='input')
    parser.add_argument('--name',default='')
    parser.add_argument('--root',default='./data3')
    parser.add_argument('--data_rpath',default='./data3')
    parser.add_argument('--I_path',default='./data3/I/')
    parser.add_argument('--V_path',default='./data3/V/')
    parser.add_argument('--anns_spath',default='./data3/')

    args=parser.parse_args()

    return args

args=parse_args()
name=args.name                   #根目录
root=args.root                   #根目录下的dataset文件夹路径
data_rpath=args.data_rpath       #图像路径
img_spath=args.I_path            #设定图像和视频的存储路径
video_img_spath=args.V_path
anns_spath=args.anns_spath       #设定annotations的存储路径
img_paths=[]
img_paths.extend(glob.glob(data_rpath+'/image/*/*.jpg'))
CLASS_DICT=collections.OrderedDict(
    {
        '短外套':1,
        '古风':2,'古装':2,
        '短裤':3,
        '短袖上衣':4,'短袖Top':4,
        '长半身裙':5,'长半身裙（到脚）':5,
        '背带裤':6,
        '长袖上衣':7,'长袖Top':7,
        '长袖连衣裙':8,
        '短马甲':9,
        '短裙':10,
        '背心上衣':11,
        '短袖连衣裙':12,
        '长袖衬衫':13,
        '中等半身裙':14,'中等半身裙（及膝）':14,
        '无袖上衣':15,
        '长外套':16,'长款外套':16,
        '无袖连衣裙':17,
        '连体衣':18,
        '长马甲':19,
        '长裤':20,
        '吊带上衣':21,'吊带Top':21,
        '中裤':22,
        '短袖衬衫':23 })
print("开始对图像库数据集进行标注文件的准备:")
images = []
annotations = []
categories = []
img_id = 0

# 更新categories
for k in list(CLASS_DICT.keys()):
    categories.append({"id": CLASS_DICT[k], "name": k})

for ip in tqdm(img_paths):
    img = cv2.imread(ip)
    h, w, _ = img.shape
    #del img
    # 获取图像路径ip对应的标注路径ap
    ap = ip.replace('image', 'image_annotation')
    ap = ap.replace('jpg', 'json')

    with open(ap, 'r') as json_f:
        img_ann = json.load(json_f)

    # 若标注为空
    if len(img_ann['annotations']) == 0:
        pass
    # 若存在标注
    else:
        # 更新images
        file_name = 'i_' + str(img_ann['img_name'][:-4]) + '_' + str(img_ann['item_id'] + '.jpg')
        #file_name=ip
        # # 保存图片至images文件夹
        print(img_spath + file_name)
        cv2.imwrite(img_spath + file_name, img)
        # del img

        img_id += 1
        images.append({'file_name': file_name,
                       'id': img_id,
                       'height': h,
                       'width': w})
        # 更新annotations
        for ann in img_ann['annotations']:
            xmin = float(ann['box'][0])
            ymin = float(ann['box'][1])
            box_w = float(ann['box'][2] - ann['box'][0] + 1)
            box_h = float(ann['box'][3] - ann['box'][1] + 1)
            #print(ann['label'])
            cls_id = CLASS_DICT[ann['label']]
            print(cls_id)
            annotations.append({'image_id': img_id,
                                'bbox': [xmin, ymin, box_w, box_h],
                                'category_id': cls_id,
                                'instance_id': ann['instance_id'],
                                'area': int(box_w * box_h),
                                'iscrowd': 0,
                                'segmentation':
                                #[np.array([xmin, ymin]), np.array([xmin+box_w, ymin]), 
                                #np.array([xmin, ymin+box_h]), np.array([xmin+box_w, ymin+box_h])]
                                [[xmin, ymin,xmin+box_w, ymin,xmin, ymin+box_h,xmin+box_w, ymin+box_h]]
                                ,
                                })

print('Finish preparing item images!')
print("Frame image starts ‘id' from ", img_id)

#del img_paths

video_paths = []  # 所有视频的路径
# video_ann_paths = [] # 所有视频标注的路径
#print(img_paths)

# for p in img_paths:
#     print(p)
#     print(glob.glob(p + '/video/*.mp4'))
#     video_paths.extend(glob.glob(p + '/video/*.mp4'))
print(glob.glob(data_rpath+'/video/*.mp4'))
video_paths.extend(glob.glob(data_rpath+'/video/*.mp4'))

print("开始对视频库直播切片进行标注文件的准备：")
print(video_paths)

def get_frame_img(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    _, frame_img = cap.read()
    cap.release()
    return frame_img


for vp in tqdm(video_paths):
    # 获取视频路径p对应的标注路径vap
    vap = vp.replace('video', 'video_annotation')
    vap = vap.replace('mp4', 'json')

    with open(vap, 'r') as json_f2:
        video_ann = json.load(json_f2)

    for frame in video_ann['frames']:
        # 如果单个frame下没有标注：
        if len(frame['annotations']) == 0:
            pass
        # 如果单个frame下有标注：
        else:
            frame_index = frame['frame_index']
            frame_img = get_frame_img(vp, frame_index)

            vh, vw, _ = frame_img.shape
            frame_img
            # 更新images
            img_id += 1
            vfile_name = 'v_' + str(video_ann['video_id']) + '_' + str(frame_index) + '.jpg'
            #vfile_name=vp
            images.append({'file_name': vfile_name,
                           'id': img_id,
                           'height': vh,
                           'width': vw})

            # # 保存图片至images文件夹
            cv2.imwrite(video_img_spath + vfile_name, frame_img)
            del frame_img

            # 更新annotations
            for fann in frame['annotations']:
                fxmin = float(fann['box'][0])
                fymin = float(fann['box'][1])
                fbox_w = float(fann['box'][2] - fann['box'][0] + 1)
                fbox_h = float(fann['box'][3] - fann['box'][1] + 1)
                fcls_id = CLASS_DICT[fann['label']]
                annotations.append({'image_id': img_id,
                                    'bbox': [fxmin, fymin, fbox_w, fbox_h],
                                    'category_id': fcls_id,
                                    'instance_id': fann['instance_id'],
                                    'area': int(fbox_w * fbox_h),
                                    'iscrowd': 0,
                                    'segmentation': [],
                                    })

print('Finish preparing frame images!')

# ‘古装’和‘古风’合为‘古风’
new_categories = [categories[i] for i, cat in enumerate(categories) if cat['name'] != '古装']

# 保存标注至annotations文件夹
all_anns = {"images": images, "annotations": annotations, "categories": new_categories}
#all_anns = {"images": images, "annotations": annotations}
with open(anns_spath +'data.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving ''.json')
dataset=['data']
images=[]
annotations=[]

ann_id=1
dataset=['data']
images=[]
annotations=[]

ann_id=1
for name in dataset:
    path='./data3/'+name+'.json'
    print(path)

    with open(path,'r') as f:
        ann=json.load(f)

    images +=ann['images']
    for item in ann['annotations']:
        item['id']=ann_id
        ann_id+=1

    annotations +=ann['annotations']
    print(name,len(ann['images']),len(ann['annotations']))

    i_images=[]
    i_ann=[]
    i_ids=[]
    v_images=[]
    v_ann=[]

    for item in tqdm(images):
        #if item['file_name'].startwith('i'):
        #print(item['file_name'])
        if item['file_name'][0]=='i':
            i_images.append(item)
            i_ids.append(item['id'])

        else:
            v_images.append(item)

    i_ids=set(i_ids)
    for item in tqdm(annotations):
        if item['image_id'] in i_ids:
            i_ann.append(item)
        else:
            v_ann.append(item)

    ann['images']=i_images
    ann['annotations']=i_ann
    print('images: ',len(ann['images']),len(ann['annotations']))
    with open('./data3/i_train.json','w') as f:
        json.dump(ann,f)


    ann['images']=v_images
    ann['annotations']=v_ann
    print('video: ',len(ann['images']),len(ann['annotations']))
    with open('./data3/v_train.json','w') as f:
        json.dump(ann,f)