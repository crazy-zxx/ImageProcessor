import os
import sys

import cv2
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGraphicsScene, QFileDialog, QMessageBox

from mainWindow import Ui_MainWindow
from pretreatmentWindow import Ui_Form


# 预处理窗口类
class PretreatmentWindow(QWidget, Ui_Form):
    signal = pyqtSignal(object)

    def __init__(self):
        super(PretreatmentWindow, self).__init__()
        self.setupUi(self)

        self.spinBox.valueChanged.connect(self.spinBoxChange)
        self.slider.valueChanged.connect(self.sliderChange)
        self.submitButton.clicked.connect(self.valueConfirm)

    def spinBoxChange(self):
        value = self.spinBox.value()
        self.slider.setValue(value)
        self.signal.emit(value)

    def sliderChange(self):
        value = self.slider.value()
        self.spinBox.setValue(value)
        # self.signal.emit(value)

    def valueConfirm(self):
        self.signal.emit('ok')
        self.close()


# 主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setupUi(self)

        # 图像预处理的子窗口（亮度、对比度、锐度调节窗口）
        self.__pretreatmentWindow = None

        # 打开的图片文件名
        self.__fileName = None
        # 原始图片数据矩阵
        self.__srcImageRGB = None
        # 结果图片数据矩阵
        self.__outImageRGB = None

        # 绑定窗口事件的响应函数
        self.openFileAction.triggered.connect(self.openFileAndShowImage)
        self.saveFileAction.triggered.connect(self.saveFile)
        self.saveFileAsAction.triggered.connect(self.saveFileAs)
        self.exitAppAction.triggered.connect(self.close)

        self.resetImageAction.triggered.connect(self.resetImage)

        self.grayAction.triggered.connect(self.toGrayImage)
        self.binaryAction.triggered.connect(self.toBinaryImage)
        self.reverseAction.triggered.connect(self.reverseImage)
        self.lightAction.triggered.connect(self.openLightWindow)

    # 打开文件并在主窗口中显示打开的图像
    def openFileAndShowImage(self):
        __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
        if __fileName and os.path.exists(__fileName):
            self.__fileName = __fileName
            __bgrImg = cv2.imread(self.__fileName)
            self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__drawImage(self.srcImageView, self.__srcImageRGB)
            self.__drawImage(self.outImageView, self.__srcImageRGB)

    # 在窗口中指定位置显示指定类型的图像
    def __drawImage(self, location, img):
        # RBG图
        if len(img.shape) > 2:
            # 获取行、列、通道数
            __height, __width, __channel = img.shape
            __qImg = QImage(img, __width, __height, __width * __channel, QImage.Format_RGB888)
            __qPixmap = QPixmap.fromImage(__qImg)
            __scene = QGraphicsScene()
            __scene.addPixmap(__qPixmap)
            location.setScene(__scene)
        # 灰度图、二值图
        else:
            __width, __height = img.shape
            __qImg = QImage(img, __width, __height, __width, QImage.Format_Indexed8)
            __qPixmap = QPixmap.fromImage(__qImg)
            __scene = QGraphicsScene()
            __scene.addPixmap(__qPixmap)
            location.setScene(__scene)

    # 执行保存图片文件的操作
    def __saveImg(self, fileName):
        # 已经打开了文件才能保存
        if fileName:
            __bgrImg = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            cv2.imwrite(fileName, __bgrImg)
            QMessageBox.information(self, '提示', '文件保存成功！')
        else:
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 保存文件，覆盖原始文件
    def saveFile(self):
        self.__saveImg(self.__fileName)

    # 文件另存
    def saveFileAs(self):
        # 已经打开了文件才能保存
        if self.__fileName:
            __fileName, _ = QFileDialog.getSaveFileName(self, '保存图片', 'Image', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
            self.__saveImg(__fileName)
        else:
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 重写窗口关闭事件函数，来关闭所有窗口。因为默认关闭主窗口子窗口依然存在。
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        sys.exit(0)

    # 重置图片到初始状态
    def resetImage(self):
        if self.__fileName:
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 灰度化
    def toGrayImage(self):
        # 只有RGB图才能灰度化
        if self.__fileName and len(self.__outImageRGB.shape) > 2:
            # 灰度化使得三通道RGB图变成单通道灰度图
            self.__outImageRGB = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 二值化
    def toBinaryImage(self):
        # 先灰度化
        self.toGrayImage()
        if self.__fileName:
            # 后阈值化为二值图
            _, self.__outImageRGB = cv2.threshold(self.__outImageRGB, 127, 255, cv2.THRESH_BINARY)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 反转图片颜色
    def reverseImage(self):
        if self.__fileName:
            self.__outImageRGB = cv2.bitwise_not(self.__outImageRGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 打开图像预处理的亮度调节子窗口
    def openLightWindow(self):
        self.__pretreatmentWindow = PretreatmentWindow()
        self.__pretreatmentWindow.setWindowTitle('亮度')
        self.__pretreatmentWindow.propertyLabel.setText('亮度')
        self.__pretreatmentWindow.show()
        self.__pretreatmentWindow.signal.connect(self.changeLight)

    def changeLight(self, val):
        print(val)
#         TODO


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
