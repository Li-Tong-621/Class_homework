import cv2
from PIL import Image
import matplotlib.pyplot as plt
img=Image.open('E://python_code//LITONG_WUZIQITRY//photo//3_huidu.jpg')  #打开图像
#img = cv2.imread("E:\python_code\LITONG_WUZIQITRY\3_huidu.jpg")
for i in range(19):
    for j in range(19):
        a1=i*35
        a2=i*35+35
        b1=j*35
        b2=j*35+35
        #print(a1,a2,b1,b2)
        #img1=img[a1:a2,b1:b2]  #需要保留的区域--裁剪
        box = (a1, b1, a2, b2)
        x = img.crop(box)
#参数1 是高度的范围，参数2是宽度的范围
        x.save('E://python_code//LITONG_WUZIQITRY//tp//'+str(i)+','+str(j)+".jpg")