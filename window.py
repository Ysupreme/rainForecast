# -*- coding: utf-8 -*-
# @File    : window.py
# @Software: PyCharm
# @Brief   : 图形化界面

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import shutil
from radar_predict import *

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__()
        self.setWindowTitle("定量降水估测系统")
        self.resize(1000, 600)
        self.init_GUI()


    # 界面初始化，设置界面布局
    def init_GUI(self):
        font = QFont('楷体', 15)

        # about_title = QLabel('欢迎使用定量降水估测系统')
        # about_title.setFont(QFont('楷体', 18))
        # about_title.show()
        # about_title.setAlignment(Qt.AlignCenter)

        button_load = QPushButton('上传雷达回波', self)
        button_load.setGeometry(20, 160, 120, 60)
        button_load.clicked.connect(self.click_button_load)

        button_predict = QPushButton('点击实现预测', self)
        button_predict.setGeometry(20, 380, 120, 60)
        button_predict.clicked.connect(self.click_button_predict)

        radar_pic = '1.jpg'
        pixmap = QPixmap(radar_pic)  # 按指定路径找到图片
        label1 = QLabel(self)
        label1.setGeometry(300, 100, 400, 400)
        label1.show()
        label1.setPixmap(pixmap)  # 在label上显示图片
        label1.setScaledContents(True)  # 让图片自适应label大小

    # 上传文件
    def click_button_load(self):
        global flag
        flag = 0
        sender = self.sender()
        print(sender.text() + '被点击')
        openfile_name = QFileDialog.getOpenFileName(self, 'chose files', '',
                                                    'grd files(*.grd)')  # 打开文件选择框选择文件
        grd_name = openfile_name[0]  # 获取图片名称
        print(grd_name)
        file_name = grd_name[-24:]
        print(file_name)
        if grd_name == '':
            pass
        else:
            target_grd_name = "C:/Users/84342/PycharmProjects/PrecipitationNowcasting/GUI_grd/" + str(file_name)  # 将图片移动到当前目录
            shutil.copy(grd_name, target_grd_name)
            flag += 1
        print('已上传', flag, '个grd文件')

    # 实现预测
    def click_button_predict(self):
        pic_path = radar_predict()
        # print(pic_path)
        pixmap = QPixmap(pic_path)  # 按指定路径找到图片
        label_pic = QLabel(self)
        label_pic.setGeometry(300, 100, 400, 400)
        label_pic.show()
        label_pic.setPixmap(pixmap)  # 在label上显示图片
        label_pic.setScaledContents(True)  # 让图片自适应label大小

    # 界面关闭事件，询问用户是否关闭
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

