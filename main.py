import os
import sys

import cv2
import numpy
from matplotlib import pyplot as plt
from PyQt5 import QtGui
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGraphicsScene, QFileDialog, QMessageBox

from mainWindow import Ui_MainWindow
from propertyWindow import Ui_Form


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
        self.openFileAction.triggered.connect(self.__openFileAndShowImage)
        self.saveFileAction.triggered.connect(self.saveFile)
        self.saveFileAsAction.triggered.connect(self.saveFileAs)
        self.exitAppAction.triggered.connect(self.close)

        self.resetImageAction.triggered.connect(self.__resetImage)

        self.grayAction.triggered.connect(self.__toGrayImage)
        self.binaryAction.triggered.connect(self.__toBinaryImage)
        self.reverseAction.triggered.connect(self.__reverseImage)
        self.lightAction.triggered.connect(self.__openLightWindow)
        self.contrastAction.triggered.connect(self.__openContrastWindow)
        self.sharpAction.triggered.connect(self.__openSharpWindow)
        self.saturationAction.triggered.connect(self.__openSaturationWindow)

        self.imageAddAction.triggered.connect(self.__addImage)
        self.imageSubtractAction.triggered.connect(self.__subtractImage)
        self.imageMultiplyAction.triggered.connect(self.__multiplyImage)
        self.zoomAction.triggered.connect(self.__openZoomWindow)
        self.rotateAction.triggered.connect(self.__openRotateWindow)

        self.histogramAction.triggered.connect(self.__histogram)
        self.histogramEqAction.triggered.connect(self.__histogramEqualization)

        self.aboutAction.triggered.connect(self.__aboutAuthor)

    # 打开文件并在主窗口中显示打开的图像
    def __openFileAndShowImage(self):
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
        # 灰度图、二值图
        else:
            __height, __width = img.shape
            __qImg = QImage(img, __width, __height, __width, QImage.Format_Indexed8)
        # 转换图片格式显示
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
    def __resetImage(self):
        if self.__fileName:
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # -----------------------------------图像预处理-----------------------------------
    # 灰度化
    def __toGrayImage(self):
        # 只有RGB图才能灰度化
        if self.__fileName and len(self.__outImageRGB.shape) > 2:
            # 灰度化使得三通道RGB图变成单通道灰度图
            self.__outImageRGB = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2GRAY)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 二值化
    def __toBinaryImage(self):
        # 先灰度化
        self.__toGrayImage()
        if self.__fileName:
            # 后阈值化为二值图
            _, self.__outImageRGB = cv2.threshold(self.__outImageRGB, 127, 255, cv2.THRESH_BINARY)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 反转图片颜色
    def __reverseImage(self):
        if self.__fileName:
            self.__outImageRGB = cv2.bitwise_not(self.__outImageRGB)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 打开属性调节子窗口（亮度、对比度、锐度、饱和度、缩放、旋转）
    def __openPropertyWindow(self, propertyName, func):
        if self.__fileName:
            self.__pretreatmentWindow = PretreatmentWindow()
            # 设置窗口内容
            self.__pretreatmentWindow.setWindowTitle(propertyName)
            self.__pretreatmentWindow.propertyLabel.setText(propertyName)
            # 设置主窗口接收子窗口发送的信号的处理函数
            self.__pretreatmentWindow.signal.connect(func)
            # 显示子窗口
            self.__pretreatmentWindow.show()

    # 打开图像预处理的亮度调节子窗口
    def __openLightWindow(self):
        self.__openPropertyWindow('亮度', self.__changeLight)

    # 打开图像预处理的对比度调节子窗口
    def __openContrastWindow(self):
        self.__openPropertyWindow('对比度', self.__changeContrast)

    # 打开图像预处理的锐度调节子窗口
    def __openSharpWindow(self):
        self.__openPropertyWindow('锐度', self.__changeSharp)

    # 打开图像预处理的饱和度调节子窗口
    def __openSaturationWindow(self):
        self.__openPropertyWindow('饱和度', self.__changeSaturation)

    # 预处理信号
    def __dealSignal(self, val):
        # 拷贝后修改副本
        __img = self.__outImageRGB.copy()
        # 如果是灰度图要转为RGB图
        if len(__img.shape) < 3:
            __img = cv2.cvtColor(__img, cv2.COLOR_GRAY2RGB)

        value = str(val)
        # 确认修改
        if value == 'ok':
            # 将暂存的修改保存为结果
            self.__outImageRGB = self.__tempImageRGB.copy()
            return None
        # 修改完成（确认已经做的修改或取消了修改）
        elif value == 'close':
            # 重绘修改预览
            self.__drawImage(self.outImageView, self.__outImageRGB)
            return None
        # 暂时修改
        else:
            return __img

    # 改变亮度或对比度
    # g(i,j)=αf(i,j)+(1-α)black+β，α用来调节对比度, β用来调节亮度
    def __lightAndContrast(self, img, alpha, beta):
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
    def __changeLight(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if numpy.size(__img) > 1:
            beta = int(val) * (255 / 100)
            # 暂存本次修改
            self.__tempImageRGB = self.__lightAndContrast(__img, 1, beta)

    # 修改对比度
    def __changeContrast(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if numpy.size(__img) > 1:
            k = int(val)
            if k != -100:
                alpha = (k + 100) / 100
            else:
                alpha = 0.01
            # 暂存本次修改
            self.__tempImageRGB = self.__lightAndContrast(__img, alpha, 0)

    # 修改锐度
    def __changeSharp(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if numpy.size(__img) > 1:
            # 比例
            k = int(val) * 0.01
            # 卷积核
            kernel = numpy.array([[0, -1, 0], [-1, 5 + k, -1], [0, -1, 0]])
            # 通过卷积实现锐化,暂存修改数据
            self.__tempImageRGB = cv2.filter2D(__img, -1, kernel)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改饱和度
    def __changeSaturation(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if numpy.size(__img) > 1:
            # 转换颜色空间到HLS
            __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
            # 比例
            k = int(val) * (255 / 100)
            # 切片修改S分量，并限制色彩数值在0-255之间
            __img[:, :, 2] = numpy.clip(__img[:, :, 2] + k, 0, 255)
            # 暂存修改数据
            self.__tempImageRGB = cv2.cvtColor(__img, cv2.COLOR_HLS2RGB)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # -----------------------------------图像运算-----------------------------------
    # 加、减、乘操作
    def __operation(self, func):
        if self.__fileName:
            __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
            if __fileName and os.path.exists(__fileName):
                __bgrImg = cv2.imread(__fileName)
                # 图片尺寸相同才能进行运算
                if self.__outImageRGB.shape == __bgrImg.shape:
                    __rgbImg = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
                    self.__outImageRGB = func(self.__outImageRGB, __rgbImg)
                    self.__drawImage(self.outImageView, self.__outImageRGB)
                else:
                    QMessageBox.information(None, '提示', '图像尺寸不一致，无法进行操作！')

    # 加
    def __addImage(self):
        self.__operation(cv2.add)

    # 减
    def __subtractImage(self):
        self.__operation(cv2.subtract)

    # 乘
    def __multiplyImage(self):
        self.__operation(cv2.multiply)

    # 打开图像预处理的缩放调节子窗口
    def __openZoomWindow(self):
        self.__openPropertyWindow('缩放', self.__changeZoom)

    # 缩放
    def __changeZoom(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if numpy.size(__img) > 1:
            # 计算比例
            i = int(val)
            if i == -100:
                k = 0.01
            elif i >= 0:
                k = (i + 10) / 10
            else:
                k = (i + 100) / 100
            # 直接cv2.resize()缩放
            self.__tempImageRGB = cv2.resize(__img, None, fx=k, fy=k, interpolation=cv2.INTER_LINEAR)
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 打开图像预处理的旋转调节子窗口
    def __openRotateWindow(self):
        self.__openPropertyWindow('旋转', self.__changeRotate)
        # 重设属性值取值范围
        self.__pretreatmentWindow.slider.setMaximum(360)
        self.__pretreatmentWindow.slider.setMinimum(-360)
        self.__pretreatmentWindow.spinBox.setMaximum(360)
        self.__pretreatmentWindow.spinBox.setMinimum(-360)

    # 旋转
    def __changeRotate(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        # None的size是1 ！！！   why？？？
        if numpy.size(__img) > 1:
            # 比例
            k = int(val)
            (h, w) = __img.shape[:2]
            (cX, cY) = (w // 2, h // 2)
            # 绕图片中心旋转
            m = cv2.getRotationMatrix2D((cX, cY), k, 1.0)
            # 计算调整后的图片显示大小，使得图片不会被切掉边缘
            cos = numpy.abs(m[0, 0])
            sin = numpy.abs(m[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))
            m[0, 2] += (nW / 2) - cX
            m[1, 2] += (nH / 2) - cY
            # 变换，并设置旋转调整后产生的无效区域为白色
            self.__tempImageRGB = __img = cv2.warpAffine(__img, m, (nW, nH), borderValue=(255, 255, 255))
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # -----------------------------------直方图均衡-----------------------------------
    # 归一化直方图
    def __histogram(self):
        if self.__fileName:
            color = {'r', 'g', 'b'}
            # 使用 matplotlib 的绘图功能同时绘制多通道 RGB 的直方图
            for i, col in enumerate(color):
                __hist = cv2.calcHist([self.__outImageRGB], [i], None, [256], [0, 256])
                plt.plot(__hist, color=col)
                plt.xlim([0, 256])
            plt.show()

    # 直方图均衡化
    def __histogramEqualization(self):
        if self.__fileName:
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                __histEq = cv2.equalizeHist(self.__outImageRGB)
                self.__outImageRGB = numpy.hstack((self.__outImageRGB, __histEq))
            # 如果是RGB图
            else:
                # 分解通道，各自均衡化，再合并通道
                (r, g, b) = cv2.split(self.__outImageRGB)
                rh = cv2.equalizeHist(r)
                gh = cv2.equalizeHist(g)
                bh = cv2.equalizeHist(b)
                self.__outImageRGB = cv2.merge((rh, gh, bh))
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 关于作者
    def __aboutAuthor(self):
        QMessageBox.information(None, '关于作者', '图像处理软件2.0\n\nCopyright © 2021–2099 赵相欣\n\n保留一切权利')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
