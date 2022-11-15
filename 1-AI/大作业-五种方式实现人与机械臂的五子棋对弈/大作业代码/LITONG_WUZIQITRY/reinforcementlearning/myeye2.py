import cv2
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import os
from PIL import Image
import matplotlib.pyplot as plt
import joblib

class myeye():
    def pho():
        # 图像预处理+
        cap = cv2.VideoCapture(1)
        cap.set(3, 1920)  # 设置分辨率
        cap.set(4, 1080)
        while (1):
            sucess, img = cap.read()
            cv2.imshow("capture", img)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                # 存储图片
                cap.release()
                cv2.imwrite("E:\\python_code\\LITONG_WUZIQITRY\\photo\\6.jpg", img)
                cv2.destroyAllWindows()
                break
    def con():
        pic_file = r'E:\\python_code\\LITONG_WUZIQITRY\\photo\\6.jpg'

        im_bgr = cv2.imread(pic_file)  # 读入图像
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)  # 转灰度
        im_gray = cv2.GaussianBlur(im_gray, (3, 3), 0)  # 滤波降噪
        im_edge = cv2.Canny(im_gray, 30, 50)  # 边缘检测
        # cv2.imshow('Go', im_edge) # 显示边缘检测结果
        cv2.imwrite('E:\\python_code\\LITONG_WUZIQITRY\\photo\\1_bianyuan.jpg', im_edge)


        # 识别并定位棋盘
        contours, hierarchy = cv2.findContours(im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取轮廓
        rect, area = None, 0  # 找到的最大四边形及其面积
        for item in contours:
            hull = cv2.convexHull(item)  # 寻找凸包
            epsilon = 0.1 * cv2.arcLength(hull, True)  # 忽略弧长10%的点
            approx = cv2.approxPolyDP(hull, epsilon, True)  # 将凸包拟合为多边形
            if len(approx) == 4 and cv2.isContourConvex(approx):  # 如果是凸四边形
                ps = np.reshape(approx, (4, 2))
                ps = ps[np.lexsort((ps[:, 0],))]
                lt, lb = ps[:2][np.lexsort((ps[:2, 1],))]
                rt, rb = ps[2:][np.lexsort((ps[2:, 1],))]
                a = cv2.contourArea(approx)  # 计算四边形面积
                if a > area:
                    area = a
                    rect = (lt, lb, rt, rb)
        if rect is None:
            print('在图像文件中找不到棋盘！')
        else:
            print('棋盘坐标：')
            print('\t左上角：(%d,%d)' % (rect[0][0], rect[0][1]))
            print('\t左下角：(%d,%d)' % (rect[1][0], rect[1][1]))
            print('\t右上角：(%d,%d)' % (rect[2][0], rect[2][1]))
            print('\t右下角：(%d,%d)' % (rect[3][0], rect[3][1]))

        im = np.copy(im_bgr)
        for p in rect:
            im = cv2.line(im, (p[0] - 10, p[1]), (p[0] + 10, p[1]), (0, 0, 255), 1)
            im = cv2.line(im, (p[0], p[1] - 10), (p[0], p[1] + 10), (0, 0, 255), 1)
        # cv2.imshow('go', im)
        cv2.imwrite('E:\\python_code\\LITONG_WUZIQITRY\\photo\\2_biaozhu.jpg', im)

        # 透视变换
        lt, lb, rt, rb = rect
        pts1 = np.float32([(10, 10), (10, 650), (650, 10), (650, 650)])  # 预期的棋盘四个角的坐标
        pts2 = np.float32([lt, lb, rt, rb])  # 当前找到的棋盘四个角的坐标
        m = cv2.getPerspectiveTransform(pts2, pts1)  # 生成透视矩阵
        board_gray = cv2.warpPerspective(im_gray, m, (660, 660))  # 对灰度图执行透视变换
        board_bgr = cv2.warpPerspective(im_bgr, m, (660, 660))  # 对彩色图执行透视变换
        # cv2.imshow('go', board_gray)
        cv2.imwrite('E:\\python_code\\LITONG_WUZIQITRY\\photo\\3_huidu.jpg', board_gray)


        #########################机器学习
        #############################
       # X = []
        #Y = []

        #for i in range(1, 390):
        #    image = cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//0//0 (' + str(i) + ').jpg')
        #    hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        #    X.append(((hist / 255).flatten()))
        #    Y.append(0)
        #for i in range(1, 32):
        #    image = cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//1//1 (' + str(i) + ').jpg')
        #    hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        #    X.append(((hist / 255).flatten()))
        #    Y.append(1)
        #for i in range(1, 39):
        #    image = cv2.imread('E://python_code//LITONG_WUZIQITRY//photo_train//2//2 (' + str(i) + ').jpg')
        #    hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
        #    X.append(((hist / 255).flatten()))
        #    Y.append(2)
        #X = np.array(X)
        #Y = np.array(Y)
        # 切分训练集和测试集
        # print(X.shape,Y.shape)
        #X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
        # 随机率为100%选取其中的30%作为测试集
        #clf0 = KNeighborsClassifier(n_neighbors=10).fit(X_train, y_train)
        ###################################
        #joblib.dump(clf0, 'shibie.pkl')
        # load model

        clf0 = joblib.load('shibie.pkl')

        img = Image.open('E://python_code//LITONG_WUZIQITRY//photo//3_huidu.jpg')  # 打开图像
        # img = cv2.imread("E:\python_code\LITONG_WUZIQITRY\3_huidu.jpg")
        for i in range(19):
            for j in range(19):
                a1 = i * 35
                a2 = i * 35 + 35
                b1 = j * 35
                b2 = j * 35 + 35
                # print(a1,a2,b1,b2)
                # img1=img[a1:a2,b1:b2]  #需要保留的区域--裁剪
                box = (a1, b1, a2, b2)
                x = img.crop(box)
                # 参数1 是高度的范围，参数2是宽度的范围
                x.save('E://python_code//LITONG_WUZIQITRY//tp//' + str(i) + ',' + str(j) + ".jpg")
        #############################
        #############################

        file1 = 'oldboard.txt'
        file2 = 'newboard.txt'
        oldboard = np.loadtxt(file2, dtype=np.int, delimiter=',', unpack=False)

        newboard=oldboard.copy()
        for i in range(19):
            for j in range(19):
                image = cv2.imread('E://python_code//LITONG_WUZIQITRY//tp//' + str(i) + ',' + str(j) + ".jpg")
                hist = cv2.calcHist([image], [0, 1], None, [256, 256], [0.0, 255.0, 0.0, 255.0])
                # hist=np.array(((hist / 255).flatten()))
                hist = [(hist / 255).flatten()]
                hist = np.array(hist)
                p = clf0.predict(hist)
                newboard[i][j]=int(p)
                if int(p)!=0:
                    print(i,j,int(p))

        np.savetxt(file1, oldboard, fmt='%d', delimiter=',')
        #把原来的棋盘变成旧棋盘
        #新识别的phase 作为新棋盘
        #print(newboard)
        np.savetxt(file2,newboard,fmt='%d',delimiter=',')
