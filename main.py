import os
import sys

import cv2
import numpy
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

    def valueConfirm(self):
        self.signal.emit('ok')
        self.close()

    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        self.signal.emit('close')


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
        # 暂存未确认保存的修改数据矩阵
        self.__tempImageRGB = None

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
        self.contrastAction.triggered.connect(self.openContrastWindow)
        self.sharpAction.triggered.connect(self.openSharpWindow)
        self.saturationAction.triggered.connect(self.openSaturationWindow)

    # 打开文件并在主窗口中显示打开的图像
    def openFileAndShowImage(self):
        __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
        if __fileName and os.path.exists(__fileName):
            self.__fileName = __fileName
            __bgrImg = cv2.imread(self.__fileName)
            self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__tempImageRGB = self.__srcImageRGB.copy()
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
            __height, __width = img.shape
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

    # -----------------------------------图像预处理-----------------------------------
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
        if self.__fileName:
            self.__pretreatmentWindow = PretreatmentWindow()
            self.__pretreatmentWindow.setWindowTitle('亮度')
            self.__pretreatmentWindow.propertyLabel.setText('亮度')
            self.__pretreatmentWindow.show()
            self.__pretreatmentWindow.signal.connect(self.changeLight)

    # 打开图像预处理的对比度调节子窗口
    def openContrastWindow(self):
        if self.__fileName:
            self.__pretreatmentWindow = PretreatmentWindow()
            self.__pretreatmentWindow.setWindowTitle('对比度')
            self.__pretreatmentWindow.propertyLabel.setText('对比度')
            self.__pretreatmentWindow.show()
            self.__pretreatmentWindow.signal.connect(self.changeContrast)

    # 打开图像预处理的锐度调节子窗口
    def openSharpWindow(self):
        if self.__fileName:
            self.__pretreatmentWindow = PretreatmentWindow()
            self.__pretreatmentWindow.setWindowTitle('锐度')
            self.__pretreatmentWindow.propertyLabel.setText('锐度')
            self.__pretreatmentWindow.show()
            self.__pretreatmentWindow.signal.connect(self.changeSharp)

    # 打开图像预处理的饱和度调节子窗口
    def openSaturationWindow(self):
        if self.__fileName:
            self.__pretreatmentWindow = PretreatmentWindow()
            self.__pretreatmentWindow.setWindowTitle('饱和度')
            self.__pretreatmentWindow.propertyLabel.setText('饱和度')
            self.__pretreatmentWindow.show()
            self.__pretreatmentWindow.signal.connect(self.changeSaturation)

    # 改变亮度或对比度
    # g(i,j)=αf(i,j)+(1-α)black+β，α用来调节对比度, β用来调节亮度
    def lightAndContrast(self, img, alpha, beta):
        if len(img.shape) < 3:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        rows, cols, channels = img.shape
        # 新建全零(黑色)图片数组
        blank = numpy.zeros([rows, cols, channels], img.dtype)
        # 计算两个图像阵列的加权和
        img = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)
        # 显示修改数据
        self.__drawImage(self.outImageView, img)
        return img

    # 修改亮度
    def changeLight(self, val):
        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        value = str(val)

        # 确认修改
        if value == 'ok':
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == 'close':
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
        # 暂时修改
        else:
            beta = int(value) * (255 / 100)
            # 暂存本次修改
            self.__tempImageRGB = self.lightAndContrast(__img, 1, beta)

    # 修改对比度
    def changeContrast(self, val):
        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        value = str(val)

        # 确认修改
        if value == 'ok':
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == 'close':
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
        # 暂时修改
        else:
            k = int(value)
            if k != -100:
                alpha = (k + 100) / 100
            else:
                alpha = 0.01
            # 暂存本次修改
            self.__tempImageRGB = self.lightAndContrast(__img, alpha, 0)

    # 修改锐度
    def changeSharp(self, val):

        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        if len(__img.shape) < 3:
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2RGB)
        value = str(val)

        # 确认修改
        if value == 'ok':
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == 'close':
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
        # 暂时修改
        else:
            k = int(value) * 0.01
            # 卷积核
            kernel = numpy.array([[0, -1, 0], [-1, 5 + k, -1], [0, -1, 0]])
            # 通过卷积实现锐化,暂存修改数据
            self.__tempImageRGB = cv2.filter2D(__img, -1, kernel)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改饱和度
    def changeSaturation(self, val):
        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        if len(__img.shape) < 3:
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2RGB)
        __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
        value = str(val)

        # 确认修改
        if value == 'ok':
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == 'close':
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
        # 暂时修改
        else:
            # 比例
            k = int(value) * (255 / 100)
            # 切片修改S分量，并限制色彩数值在0-255之间
            __img[:, :, 2] = numpy.clip(__img[:, :, 2] + k, 0, 255)
            self.__tempImageRGB = cv2.cvtColor(__img, cv2.COLOR_HLS2RGB)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # -----------------------------------图像运算-----------------------------------
    # 加

    # 减

    # 乘

    # 打开图像预处理的对比度调节子窗口
    def openZoomWindow(self):
        if self.__fileName:
            self.__pretreatmentWindow = PretreatmentWindow()
            self.__pretreatmentWindow.setWindowTitle('缩放')
            self.__pretreatmentWindow.propertyLabel.setText('缩放')
            self.__pretreatmentWindow.show()
            self.__pretreatmentWindow.signal.connect(self.changeZoom)

    # 缩放
    def changeZoom(self):
        if self.__fileName:
            status = 'zoom'
            img = self.copyByStatus(status)
            # 计算比例
            k = self.zoomBox.value() / 100
            # 直接cv2.resize()缩放
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)

    # 旋转
    def changeRotate(self):
        if self.__fileName:
            __status = 'rotate'
            img = self.copyByStatus(__status)
            k = self.ui.rotateBox.value()

            # 计算调整后的图片显示大小，使得图片不会被切掉边缘
            (h, w) = img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # 旋转
            m = cv2.getRotationMatrix2D((cX, cY), k, 1.0)
            cos = numpy.abs(m[0, 0])
            sin = numpy.abs(m[0, 1])
            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            # adjust the rotation matrix to take into account translation
            m[0, 2] += (nW / 2) - cX
            m[1, 2] += (nH / 2) - cY
            # 变换，并设置旋转调整后产生的无效区域为白色
            img = cv2.warpAffine(img, m, (nW, nH), borderValue=(255, 255, 255))

    # -----------------------------------直方图均衡-----------------------------------

    # -----------------------------------空域滤波-----------------------------------


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
