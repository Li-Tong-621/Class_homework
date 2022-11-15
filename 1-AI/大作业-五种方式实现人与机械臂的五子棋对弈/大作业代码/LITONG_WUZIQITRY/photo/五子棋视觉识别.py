# -*- coding: utf-8 -*-
"""
识别图像中的围棋局面
"""
import cv2
import numpy as np

#from stats import show_phase, stats


class GoPhase:
    """从图片中识别围棋局面"""

    def __init__(self, pic_file, offset=3.75):
        """构造函数，读取图像文件，预处理"""

        self.offset = offset  # 黑方贴七目半
        self.im_bgr = cv2.imread(pic_file)  # 原始的彩色图像文件，BGR模式
        self.im_gray = cv2.cvtColor(self.im_bgr, cv2.COLOR_BGR2GRAY)  # 转灰度图像
        self.im_gray = cv2.GaussianBlur(self.im_gray, (3, 3), 0)  # 灰度图像滤波降噪
        self.im_edge = cv2.Canny(self.im_gray, 30, 50)  # 边缘检测获得边缘图像

        self.board_gray = None  # 棋盘灰度图
        self.board_bgr = None  # 棋盘彩色图
        self.rect = None  # 棋盘四个角的坐标，顺序为lt/lb/rt/rb
        self.phase = None  # 用以表示围棋局面的二维数组
        self.result = None  # 对弈结果

        self._find_chessboard()  # 找到棋盘
        self._location_grid()  # 定位棋盘格子
        self._identify_chessman()  # 识别棋子
        self._stats()  # 统计黑白双方棋子和围空

    def _find_chessboard(self):
        """找到棋盘"""

        contours, hierarchy = cv2.findContours(self.im_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 提取轮廓
        area = 0  # 找到的最大四边形及其面积
        for item in contours:
            hull = cv2.convexHull(item)  # 寻找凸包
            epsilon = 0.1 * cv2.arcLength(hull, True)  # 忽略弧长10%的点
            approx = cv2.approxPolyDP(hull, epsilon, True)  # 将凸包拟合为多边形

            if len(approx) == 4 and cv2.isContourConvex(approx):  # 如果是凸四边形
                ps = np.reshape(approx, (4, 2))  # 四个角的坐标
                ps = ps[np.lexsort((ps[:, 0],))]  # 排序区分左右
                lt, lb = ps[:2][np.lexsort((ps[:2, 1],))]  # 排序区分上下
                rt, rb = ps[2:][np.lexsort((ps[2:, 1],))]  # 排序区分上下

                a = cv2.contourArea(approx)
                if a > area:
                    area = a
                    self.rect = (lt, lb, rt, rb)

        if not self.rect is None:
            pts1 = np.float32([(10, 10), (10, 650), (650, 10), (650, 650)])  # 预期的棋盘四个角的坐标
            pts2 = np.float32(self.rect)  # 当前找到的棋盘四个角的坐标
            m = cv2.getPerspectiveTransform(pts2, pts1)  # 生成透视矩阵
            self.board_gray = cv2.warpPerspective(self.im_gray, m, (660, 660))  # 执行透视变换
            self.board_bgr = cv2.warpPerspective(self.im_bgr, m, (660, 660))  # 执行透视变换

    def _location_grid(self):
        """定位棋盘格子"""

        if self.board_gray is None:
            return

        circles = cv2.HoughCircles(self.board_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=90, param2=16, minRadius=10,
                                   maxRadius=20)  # 圆检测

        if circles == None:
            return 0
        xs, ys = circles[0, :, 0], circles[0, :, 1]  # 所有棋子的x坐标和y坐标
        xs.sort()
        ys.sort()

        k = 1
        while xs[k] - xs[:k].mean() < 15:
            k += 1
        x_min = int(round(xs[:k].mean()))

        k = 1
        while ys[k] - ys[:k].mean() < 15:
            k += 1

        y_min = int(round(ys[:k].mean()))

        k = -1
        while xs[k:].mean() - xs[k - 1] < 15:
            k -= 1
        x_max = int(round(xs[k:].mean()))

        k = -1
        while ys[k:].mean() - ys[k - 1] < 15:
            k -= 1
        y_max = int(round(ys[k:].mean()))

        if abs(600 - (x_max - x_min)) < abs(600 - (y_max - y_min)):
            v_min, v_max = x_min, x_max
        else:
            v_min, v_max = y_min, y_max

        pts1 = np.float32([[22, 22], [22, 598], [598, 22], [598, 598]])  # 棋盘四个角点的最终位置
        pts2 = np.float32([(v_min, v_min), (v_min, v_max), (v_max, v_min), (v_max, v_max)])
        m = cv2.getPerspectiveTransform(pts2, pts1)
        self.board_gray = cv2.warpPerspective(self.board_gray, m, (620, 620))
        self.board_bgr = cv2.warpPerspective(self.board_bgr, m, (620, 620))

    def _identify_chessman(self):
        """识别棋子"""

        if self.board_gray is None:
            return

        mesh = np.linspace(22, 598, 19, dtype=np.int)
        rows, cols = np.meshgrid(mesh, mesh)

        circles = cv2.HoughCircles(self.board_gray, cv2.HOUGH_GRADIENT, 1, 20, param1=40, param2=10, minRadius=12,
                                   maxRadius=18)
        circles = np.uint32(np.around(circles[0]))

        self.phase = np.zeros_like(rows, dtype=np.uint8)
        im_hsv = cv2.cvtColor(self.board_bgr, cv2.COLOR_BGR2HSV_FULL)

        print(circles)
        print(type(circles))
        for circle in circles:
            row = int(round((circle[1] - 22) / 32))
            col = int(round((circle[0] - 22) / 32))

            print(row)
            print(type(row))
            print(col)
            print(type(col))

            hsv_ = im_hsv[cols[row, col] - 5:cols[row, col] + 5, rows[row, col] - 5:rows[row, col] + 5]
            s = np.mean(hsv_[:, :, 1])
            v = np.mean(hsv_[:, :, 2])

            if 0 < v < 115:
                self.phase[row, col] = 1  # 黑棋
            elif 0 < s < 50 and 114 < v < 256:
                self.phase[row, col] = 2  # 白棋

    def _stats(self):
        """统计黑白双方棋子和围空"""

        self.result = stats(self.phase)

    def show_image(self, name='gray', win="GoPhase"):
        """显示图像"""

        if name == 'bgr':
            im = self.board_bgr
        elif name == 'gray':
            im = self.board_gray
        else:
            im = self.im_bgr

        if im is None:
            print('识别失败，无图像可供显示')
        else:
            cv2.imshow(win, im)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def show_phase(self):
        """显示局面"""

        if self.phase is None:
            print('识别失败，无围棋局面可供显示')
        else:
            show_phase(self.phase)

    def show_result(self):
        """显示结果"""

        if self.result is None:
            print('识别失败，无对弈结果可供显示')
        else:
            black, white, common = self.result
            B = black + common / 2 - self.offset
            W = white + common / 2 + self.offset
            result = '黑胜' if B > W else '白胜'

            print('黑方：%0.1f，白方：%0.1f，%s' % (B, W, result))


if __name__ == '__main__':
    go = GoPhase('00.jpg')
    go.show_image('origin')
    go.show_image('gray')
    go.show_phase()
    go.show_result()
