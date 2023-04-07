#让放在tools/gen_det_ann_fs.py
import glob
from multiprocessing import Pool, pool
import os
import collections
import shutil
from PIL import Image
import cv2
import json

#其实直接用exp3.py就行
data_rpath='./data'
img_paths=[]
img_paths.extend(glob.glob(data_rpath+'/image/*/*.jpg'))
img_spath='./data/test_dataset_fs/images/'
anns_spath='./data/test_dataset_fs/annotations/'
video_img_spath='./data/test_dataset_fs/video_images/'

if not os.path.exists(img_spath):
    os.makedirs(img_spath)
if not os.path.exists(anns_spath):
    os.makedirs(anns_spath)
if not os.path.exists(video_img_spath):
    os.makedirs(video_img_spath)   

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

print('开始对图像库数据集进行标注文件的准备：')
images=[]
annotations=[]
categories=[]
for k in list(CLASS_DICT.keys()):
    categories.append({"id": CLASS_DICT[k], "name": k})


def process_image(ips):
    img_id,ip=ips
    name=ip.split('/')[-1]
    #print(ip)
    item_id=ip.split('/')[-2]
    file_name='i_'+name[:-4]+'_'+item_id+'.jpg'
    shutil.copy(ip,img_spath+file_name)
pool=Pool(20)
pool.map(process_image,list(enumerate(img_paths)))
pool.close()
pool.join()

video_paths=[]
video_paths.extend(glob.glob(data_rpath+'/video/*.mp4'))
#print(video_paths)
def process_video(vps):
    img_id,vp=vps
    cap=cv2.VideoCapture(vp)
    video_id=vp.split('/')[-1][:-4]
    for frame in range(20,390,40):#40帧抽取一张
        frame_index=frame
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_index)
        flag,frame_img=cap.read()
        if not flag:
            cap.release()
            break

        img_id+=1
        vfile_name='v_'+video_id+'_'+str(frame_index)+'.jpg'
        #print(video_img_spath+vfile_name)
        cv2.imwrite(video_img_spath+vfile_name,frame_img)

        del frame_img
    cap.release()
#print(1)
pool=Pool(20)
pool.map(process_video,list(enumerate(video_paths)))
pool.close()
pool.join()
#print(1)
# ‘古装’和‘古风’合为‘古风’
new_categories = [categories[i] for i, cat in enumerate(categories) if cat['name'] != '古装']
img_id=0
ann_id=0
for i,name in enumerate(os.listdir(img_spath)):
    img=Image.open(os.path.join(img_spath,name))
    w,h=img.size
    images.append(
        {
            'file_name':name,
            'id':img_id,
            'height':h,
            'width':w
        }
    )
    annotations.append(
        {
            'id':ann_id,
            'image_id':img_id,
            'bbox':[0,0,10,10],
            'area':100,
            'category_id':1,
            'iscrowd':0,
            'segmentation':[],
        }
    )
    img_id+=1
    ann_id+=1
all_anns = {"images": images, "annotations": annotations, "categories": new_categories}
with open(anns_spath + 'test_image.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving train.json')

images = []
annotations = []
for i,name in enumerate(os.listdir(video_img_spath)):
    img=Image.open(os.path.join(video_img_spath,name))
    w,h=img.size
    images.append(
        {
            'file_name':name,
            'id':img_id,
            'height':h,
            'width':w
        }
    )
    annotations.append(
        {
            'id':ann_id,
            'image_id':img_id,
            'bbox':[0,0,10,10],
            'area':100,
            'category_id':1,
            'iscrowd':0,
            'segmentation':[],
        }
    )
    img_id+=1
    ann_id+=1


all_anns = {"images": images, "annotations": annotations, "categories": new_categories}
with open(anns_spath + 'test_video.json', 'w') as json_f3:
    json.dump(all_anns, json_f3)

print('Finish saving train.json')

#python mmdetection/tools/test.py exp3.py ./expresult/det/lastest.pth --format-only --options='jsonfile_prefix=./expresult/i_fs_cascade'

#python mmdetection/tools/test.py exp3.py ./expresult/det/lastest.pth --format-only --options='jsonfile_prefix=./expresult/v_fs_cascade'
