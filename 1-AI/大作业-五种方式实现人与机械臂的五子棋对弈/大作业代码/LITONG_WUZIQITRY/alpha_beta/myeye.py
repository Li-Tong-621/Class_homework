import cv2
import numpy as np

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

        mesh = np.linspace(10, 650, 19, dtype=np.int)
        rows, cols = np.meshgrid(mesh, mesh)
        circles = cv2.HoughCircles(board_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=30, minRadius=14,
                               maxRadius=16)  # 再做一次圆检测
        circles = np.uint32(np.around(circles[0]))
        phase = np.zeros_like(rows, dtype=np.uint8)
        im_hsv = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2HSV_FULL)


        file1 = 'oldboard.txt'
        file2 = 'newboard.txt'
        oldboard = np.loadtxt(file2, dtype=np.int, delimiter=',', unpack=False)

        newboard=oldboard.copy()
        for circle in circles:
            print(circle)
            row = int(round((circle[1] - 10) / 35))
            col = int(round((circle[0] - 10) / 35))
            print(row, col)
            # phase[row,col] = 1
            hsv_ = im_hsv[cols[row, col] - 5:cols[row, col] + 5, rows[row, col] - 5:rows[row, col] + 5]
            s = np.mean(hsv_[:, :, 1])
            v = np.mean(hsv_[:, :, 2])
            if 0 < v < 115:
                phase[row, col] = 1  # 黑棋
                newboard[col][row]=1
                #print(row,col)
            elif 0 < s < 50 and 114 < v < 256:
                phase[row, col] = 2  # 白棋
                newboard[col][row] = 2

        np.savetxt(file1, oldboard, fmt='%d', delimiter=',')
        #把原来的棋盘变成旧棋盘
        #新识别的phase 作为新棋盘
        #print(newboard)
        np.savetxt(file2,newboard,fmt='%d',delimiter=',')
