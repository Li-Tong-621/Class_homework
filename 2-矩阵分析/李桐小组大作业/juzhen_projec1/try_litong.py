from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import scipy.io as scio
import os
import main
from main import isClose
import numpy as np
import cv2
import drawMesh_1
import stepOne
import stepTwo
import os
class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
#处理背景
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 450)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.retranslateUi(MainWindow)
#标签
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(0, 0, 801, 451))
        self.label.setMinimumSize(QtCore.QSize(511, 341))
        self.label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("D:/文件/A大二大创/photo/tri-2.png"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
#文件按钮所在背景
        self.pushButton2 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton2.setGeometry(QtCore.QRect(0, 0, 801, 451))
        self.pushButton2.setStyleSheet("background:url(D:/文件/A大二大创/photo/sec-2.png)")
        self.pushButton2.setText("")
        self.pushButton2.setCheckable(True)
        self.pushButton2.setObjectName("pushButton2")
#文件按钮
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(190, 90, 75, 23))
        self.pushButton.setObjectName("pushButton")
        self.pushButton.setText("打开")
#最开头按钮
        self.pushButton1 = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton1.setGeometry(QtCore.QRect(0, 0, 801, 451))
        self.pushButton1.setStyleSheet("background:url(D:/文件/A大二大创/photo/first-2.png)")
        self.pushButton1.setText("")
        self.pushButton1.setObjectName("pushButton1")


        MainWindow.setCentralWidget(self.centralwidget)
        self.retranslateUi(MainWindow)
        self.pushButton1.clicked.connect(self.pushButton1.close)
        self.pushButton.pressed.connect(self.pushButton2.close)
        self.pushButton.pressed.connect(self.pushButton.close)
#'''加一个如果没选/选择的位置不对的输出'''
        self.pushButton.pressed.connect(self.openfile)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)



    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "打开多个文件"))

    def openfile(self):
        global openfile_name1
        #os.system("python GL.py")
        openfile_name1 = QFileDialog.getOpenFileNames(self, '选择文件')

        #print(type(openfile_name1))
        openfile_name1 = openfile_name1[0]
        openfile_name1=''.join(openfile_name1)
        openfile_name1=openfile_name1[28:]
        L=[]
        L.append(openfile_name1)
        L=','.join(L)
        f = open("name.txt", "w")
        f.write(L)
        f.close()

        #print(type(openfile_name))
        print(openfile_name1)
        MainWindow.close()
        os.system("main.py")
        with open('main.py', 'r') as f:
            exec(f.read())


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

