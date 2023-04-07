import json
import pandas as pd
import mmcv
#转换成coco____________________________________________________________
#______________________________________________________________________
import argparse
def parse_args():
    parser=argparse.ArgumentParser(description='input')
    parser.add_argument('--original_json_I',default='./data3/i_train.json')
    parser.add_argument('--original_json_V',default='./data3/v_train.json')
    parser.add_argument('--now_json_I',default='./expresult/i_fs_cascade.bbox.json')
    parser.add_argument('--now_json_V',default='./expresult/v_fs_cascade.bbox.json')
    parser.add_argument('--will_json_I',default='./expresult/i_fs_cascade.json')
    parser.add_argument('--will_json_V',default='./expresult/v_fs_cascade.json')

    args=parser.parse_args()

    return args

args=parse_args()
original_json_I=args.original_json_I
original_json_V=args.original_json_V
now_json_I=args.now_json_I
now_json_V=args.now_json_V
will_json_I=args.will_json_I
will_json_V=args.will_json_V



#图片
# with open(original_json_I, 'r') as f:
#     test_json_raw=json.load(f)
# test_json=json.load(open('{}'.format(now_json_I),'r'))

# test_json_raw['annotations']=test_json

# with open('{}'.format(will_json_I),'w') as fp:
#     json.dump(test_json_raw,fp)


# dt_df=pd.DataFrame(test_json_raw['annotations'])
# #print('all_images: ',len(test_json_raw['images']))
# #print('all unique image ann: ',len(dt_df['image_id'].unique()))


#视频:
with open(original_json_V, 'r') as f:
    test_json_raw=json.load(f)
#test_json_raw=json.load('i_train.json','r)
test_json=json.load(open('{}'.format(now_json_V),'r'))

test_json_raw['annotations']=test_json

with open('{}'.format(will_json_V),'w') as fp:
    json.dump(test_json_raw,fp)


dt_df=pd.DataFrame(test_json_raw['annotations'])
#print('all_images: ',len(test_json_raw['images']))
#print('all unique image ann: ',len(dt_df['image_id'].unique()))

print('5-coco!')
