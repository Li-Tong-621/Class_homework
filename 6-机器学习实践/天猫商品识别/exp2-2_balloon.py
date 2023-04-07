# 继承自一个原始配置文件
_base_ = './mmdetection/configs/mask_rcnn/mask_rcnn_r50_caffe_fpn_mstrain-poly_1x_coco.py'

# 对头中的类别数量进行修改来匹配数据集的标注
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# 修改数据集相关设置
dataset_type = 'COCODataset'
classes = ('balloon',)
data = dict(
    train=dict(
        img_prefix='./balloon/train/',
        classes=classes,
        ann_file='./balloon/train/annotation_coco.json'),
    val=dict(
        img_prefix='./balloon/val/',
        classes=classes,
        ann_file='./balloon/val/annotation_coco.json'),
    test=dict(
        img_prefix='./balloon/val/',
        classes=classes,
        ann_file='./balloon/val/annotation_coco.json'))



#load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'

