import os
import random
import sys

import cv2
import numpy
import numpy as np
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QWidget, QApplication, QGraphicsScene, QFileDialog, QMessageBox
from matplotlib import pyplot as plt

from mainWindow import Ui_MainWindow
from propertyWindow import Ui_Form


# 预处理窗口类
# 就是弹出来的调整各种属性值的小窗口
class PropertyWindow(QWidget, Ui_Form):
    # 信号只能在Object的子类中创建，并且只能在创建类的时候的时候添加，而不能等类已经定义完再作为动态属性添加进去。
    # 自定义的信号在__init__()函数之前定义。
    # 自定义一个信号signal，有一个object类型的参数
    # **** 鬼知道咋回事，想用str类型但会出bug，只好用object了 ****
    signal = QtCore.pyqtSignal(object)

    # 类初始化
    def __init__(self):
        # 调用父类的初始化
        super(PropertyWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)

        # 绑定窗口组件响应事件的处理函数（将窗口中的组件被用户触发的点击、值变化等事件绑定到处理函数）
        # 数值框的值改变
        self.spinBox.valueChanged.connect(self.__spinBoxChange)
        # 滑动条的值改变
        self.slider.valueChanged.connect(self.__sliderChange)
        # 点击确认按钮
        self.submitButton.clicked.connect(self.__valueConfirm)

    # 数值框值改变的处理函数
    def __spinBoxChange(self):
        # 获取数值框的当前值
        value = self.spinBox.value()
        # 与滑动条进行数值同步
        self.slider.setValue(value)
        # 发送信号到主窗口，参数是当前数值（主窗口有自定义的接收并处理该信号的函数）
        self.signal.emit(value)

    # 滑动条值改变的处理函数
    def __sliderChange(self):
        # 获取滑动条的当前值
        value = self.slider.value()
        # 与数值框进行数值同步
        # 注意：该操作也会触发数值框的数值改变，即会触发调用__spinBoxChange()，所以不要需要在此处重复发信号到主窗口
        self.spinBox.setValue(value)

    # 确认按钮按下的处理函数
    def __valueConfirm(self):
        # 发送确认修改信号
        self.signal.emit('ok')
        # 关闭窗口
        self.close()

    # 重写窗口关闭处理
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        # 发送取消修改信号，关闭窗口时触发
        self.signal.emit('close')


# 主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):
    # 类初始化
    def __init__(self):
        # 调用父类的初始化
        super(MainWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)

        # 图像属性调整的子窗口，初始化默认空（亮度、对比度、锐度、饱和度、色调、旋转、缩放调节窗口）
        self.__propertyWindow = None

        # 当前打开的图片文件名，初始化默认空
        self.__fileName = None

        # 保存图片原始数据，初始化默认空
        self.__srcImageRGB = None
        # 保存图片最终处理结果的数据，初始化默认空
        self.__outImageRGB = None
        # 保存图片暂时修改的数据，初始化默认空
        # （在修改图像属性未点击确认时，需要暂存修改数据，如果确认后将临时数据同步为最终结果数据，如果未确认将复原数据）
        self.__tempImageRGB = None

        # 绑定窗口事件的响应函数
        # 文件菜单
        # 打开文件
        self.openFileAction.triggered.connect(self.__openFileAndShowImage)
        # 保存文件
        self.saveFileAction.triggered.connect(self.saveFile)
        # 另存为文件
        self.saveFileAsAction.triggered.connect(self.saveFileAs)
        # 退出程序
        self.exitAppAction.triggered.connect(self.close)

        # 重置图像菜单
        # 重置图像
        self.resetImageAction.triggered.connect(self.__resetImage)

        # 直接灰度映射菜单
        # 灰度化
        self.grayAction.triggered.connect(self.__toGrayImage)
        # 二值化
        self.binaryAction.triggered.connect(self.__toBinaryImage)
        # 颜色反转
        self.reverseAction.triggered.connect(self.__reverseImage)
        # 亮度调整
        self.lightAction.triggered.connect(self.__openLightWindow)
        # 对比度调整
        self.contrastAction.triggered.connect(self.__openContrastWindow)
        # 锐度调整
        self.sharpAction.triggered.connect(self.__openSharpWindow)
        # 饱和度调整
        self.saturationAction.triggered.connect(self.__openSaturationWindow)
        # 色度调整
        self.hueAction.triggered.connect(self.__openHueWindow)

        # 图像运算菜单
        # 加
        self.imageAddAction.triggered.connect(self.__addImage)
        # 减
        self.imageSubtractAction.triggered.connect(self.__subtractImage)
        # 乘
        self.imageMultiplyAction.triggered.connect(self.__multiplyImage)
        # 缩放
        self.zoomAction.triggered.connect(self.__openZoomWindow)
        # 旋转
        self.rotateAction.triggered.connect(self.__openRotateWindow)

        # 直方图均衡菜单
        # 归一化直方图
        self.histogramAction.triggered.connect(self.__histogram)
        # 直方图均衡化
        self.histogramEqAction.triggered.connect(self.__histogramEqualization)

        # 噪声菜单
        # 加高斯噪声
        self.addGaussianNoiseAction.triggered.connect(self.__addGasussNoise)
        # 加均匀噪声
        self.addUiformNoiseAction.triggered.connect(self.__addUniformNoise)
        # 加脉冲（椒盐）噪声
        self.addImpulseNoiseAction.triggered.connect(self.__addImpulseNoise)

        # 空域滤波菜单
        # 均值滤波
        self.meanValueAction.triggered.connect(self.__meanValueFilter)
        # 中值滤波
        self.medianValueAction.triggered.connect(self.__medianValueFilter)
        # Sobel算子锐化
        self.sobelAction.triggered.connect(self.__sobel)
        # Prewitt算子锐化
        self.prewittAction.triggered.connect(self.__prewitt)
        # 拉普拉斯算子锐化
        self.laplacianAction.triggered.connect(self.__laplacian)

        # 关于菜单
        # 关于作者
        self.aboutAction.triggered.connect(self.__aboutAuthor)

    # -----------------------------------文件-----------------------------------
    # 打开文件并在主窗口中显示打开的图像
    def __openFileAndShowImage(self):
        # 打开文件选择窗口
        __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
        # 文件存在
        if __fileName and os.path.exists(__fileName):
            # 设置打开的文件名属性
            self.__fileName = __fileName
            # 转换颜色空间，cv2默认打开BGR空间，Qt界面显示需要RGB空间，所以就统一到RGB吧
            # __bgrImg = cv2.imread(self.__fileName)
            # cv2 读取不了有中文名的图像文件 ！
            # 所以用numpy读取数据，再用cv2.imdecode解码数据来解决。
            __bgrImg = cv2.imdecode(np.fromfile(self.__fileName, dtype=np.uint8), -1)
            # 设置初始化数据
            self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            self.__outImageRGB = self.__srcImageRGB.copy()
            self.__tempImageRGB = self.__srcImageRGB.copy()
            # 在窗口中左侧QGraphicsView区域显示图片
            self.__drawImage(self.srcImageView, self.__srcImageRGB)
            # 在窗口中右侧QGraphicsView区域显示图片
            self.__drawImage(self.outImageView, self.__srcImageRGB)

    # 在窗口中指定的QGraphicsView区域（左或右）显示指定类型（rgb、灰度、二值）的图像
    def __drawImage(self, location, img):
        # RBG图
        if len(img.shape) > 2:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width, __channel = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width * __channel, QImage.Format_RGB888)
        # 灰度图、二值图
        else:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width, QImage.Format_Indexed8)

        # 创建QPixmap对象
        __qPixmap = QPixmap.fromImage(__qImg)
        # 创建显示容器QGraphicsScene对象
        __scene = QGraphicsScene()
        # 填充QGraphicsScene对象
        __scene.addPixmap(__qPixmap)
        # 将QGraphicsScene对象设置到QGraphicsView区域实现图片显示
        location.setScene(__scene)

    # 执行保存图片文件的操作
    def __saveImg(self, fileName):
        # 已经打开了文件才能保存
        if fileName:
            # RGB转BRG空间后才能通过opencv正确保存
            __bgrImg = cv2.cvtColor(self.__outImageRGB, cv2.COLOR_RGB2BGR)
            # 保存
            cv2.imwrite(fileName, __bgrImg)
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存成功！')
        else:
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 保存文件，覆盖原始文件
    def saveFile(self):
        self.__saveImg(self.__fileName)

    # 文件另存
    def saveFileAs(self):
        # 已经打开了文件才能保存
        if self.__fileName:
            # 打开文件保存的选择窗口
            __fileName, _ = QFileDialog.getSaveFileName(self, '保存图片', 'Image',
                                                        'Image Files(*.png *.jpeg *.jpg *.bmp)')
            self.__saveImg(__fileName)
        else:
            # 消息提示窗口
            QMessageBox.information(self, '提示', '文件保存失败！')

    # 重写窗口关闭事件函数，来关闭所有窗口。因为默认关闭主窗口子窗口依然存在。
    def closeEvent(self, a0: QtGui.QCloseEvent) -> None:
        sys.exit(0)

    # -----------------------------------重置图片-----------------------------------
    # 重置图片到初始状态
    def __resetImage(self):
        if self.__fileName:
            # 还原文件打开时的初始化图片数据
            self.__outImageRGB = self.__srcImageRGB.copy()
            # 窗口显示图片
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

    # 执行打开属性调节子窗口（亮度、对比度、锐度、饱和度、色调、缩放、旋转）
    def __openPropertyWindow(self, propertyName, func):
        if self.__fileName:
            if self.__propertyWindow:
                self.__propertyWindow.close()
            self.__propertyWindow = PropertyWindow()
            # 设置窗口内容
            self.__propertyWindow.setWindowTitle(propertyName)
            self.__propertyWindow.propertyLabel.setText(propertyName)
            # 接收信号
            # 设置主窗口接收子窗口发送的信号的处理函数
            self.__propertyWindow.signal.connect(func)
            # 禁用主窗口菜单栏，子窗口置顶，且无法切换到主窗口
            self.__propertyWindow.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)
            self.__propertyWindow.setWindowModality(QtCore.Qt.ApplicationModal)
            # 显示子窗口
            self.__propertyWindow.show()

    # 亮度调节子窗口
    def __openLightWindow(self):
        self.__openPropertyWindow('亮度', self.__changeLight)

    # 对比度调节子窗口
    def __openContrastWindow(self):
        self.__openPropertyWindow('对比度', self.__changeContrast)

    # 锐度调节子窗口
    def __openSharpWindow(self):
        self.__openPropertyWindow('锐度', self.__changeSharp)

    # 饱和度调节子窗口
    def __openSaturationWindow(self):
        self.__openPropertyWindow('饱和度', self.__changeSaturation)

    # 色调调节子窗口
    def __openHueWindow(self):
        self.__openPropertyWindow('色调', self.__changeHue)

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

    # 执行改变亮度或对比度
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
        # None的size是1 ！！！
        if numpy.size(__img) > 1:
            beta = int(val) * (255 / 100)
            # 暂存本次修改
            self.__tempImageRGB = self.__lightAndContrast(__img, 1, beta)

    # 修改对比度
    def __changeContrast(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
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
        if numpy.size(__img) > 1:
            # 比例
            k = int(val) * 0.01
            if k != 0:
                # 卷积核（拉普拉斯算子）
                kernel = numpy.array([[-1, -1, -1], [-1, 9 + k, -1], [-1, -1, -1]])
                # 通过卷积实现锐化,暂存修改数据
                self.__tempImageRGB = cv2.filter2D(__img, -1, kernel)
            else:
                self.__tempImageRGB = self.__outImageRGB.copy()
            # 显示修改数据
            self.__drawImage(self.outImageView, self.__tempImageRGB)

    # 修改饱和度
    def __changeSaturation(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
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

    # 修改色调
    # OpenCV中hue通道的取值范围是0 - 180
    def __changeHue(self, val):
        # 预处理接收到的信号
        __img = self.__dealSignal(val)
        # 如果修改了属性值
        if numpy.size(__img) > 1:
            # 转换颜色空间到HLS
            __img = cv2.cvtColor(__img, cv2.COLOR_RGB2HLS)
            # 比例
            k = int(val) * (90 / 100)
            # 切片修改H分量，并限制色彩数值在0-180之间
            __img[:, :, 0] = (__img[:, :, 0] + k) % 180
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
                # __bgrImg = cv2.imread(__fileName)
                __bgrImg = cv2.imdecode(np.fromfile(__fileName, dtype=np.uint8), -1)
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

    # 缩放调节子窗口
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

    # 旋转调节子窗口
    def __openRotateWindow(self):
        self.__openPropertyWindow('旋转', self.__changeRotate)
        if self.__fileName:
            # 重设属性值取值范围
            self.__propertyWindow.slider.setMaximum(360)
            self.__propertyWindow.slider.setMinimum(-360)
            self.__propertyWindow.spinBox.setMaximum(360)
            self.__propertyWindow.spinBox.setMinimum(-360)

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
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                # __hist = cv2.calcHist([self.__outImageRGB], [0], None, [256], [0, 256])
                # __hist /= self.__outImageRGB.shape[0] * self.__outImageRGB.shape[1]
                # plt.plot(__hist)
                # 使用 matplotlib 的绘图功能同时绘制单通道的直方图
                # density的类型是 bool型，指定为True,则为频率直方图，反之为频数直方图
                plt.hist(self.__outImageRGB.ravel(), bins=255, rwidth=0.8, range=(0, 256), density=True)
            # 如果是RGB图
            else:
                color = {'r', 'g', 'b'}
                # 使用 matplotlib 的绘图功能同时绘制多通道 RGB 的直方图
                for i, col in enumerate(color):
                    __hist = cv2.calcHist([self.__outImageRGB], [i], None, [256], [0, 256])
                    __hist /= self.__outImageRGB.shape[0] * self.__outImageRGB.shape[1]
                    plt.plot(__hist, color=col)
            # x轴长度区间
            plt.xlim([0, 256])
            # 显示直方图
            plt.show()

    # 直方图均衡化
    def __histogramEqualization(self):
        if self.__fileName:
            # 如果是灰度图
            if len(self.__outImageRGB.shape) < 3:
                self.__outImageRGB = cv2.equalizeHist(self.__outImageRGB)
            # 如果是RGB图
            else:
                # 分解通道，各自均衡化，再合并通道
                (r, g, b) = cv2.split(self.__outImageRGB)
                rh = cv2.equalizeHist(r)
                gh = cv2.equalizeHist(g)
                bh = cv2.equalizeHist(b)
                self.__outImageRGB = cv2.merge((rh, gh, bh))
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # -----------------------------------噪声-----------------------------------
    # 加高斯噪声
    def __addGasussNoise(self):
        if self.__fileName:
            # 图片灰度标准化
            self.__outImageRGB = numpy.array(self.__outImageRGB / 255, dtype=float)
            # 产生高斯噪声
            noise = numpy.random.normal(0, 0.001 ** 0.5, self.__outImageRGB.shape)
            # 叠加图片和噪声
            out = cv2.add(self.__outImageRGB, noise)
            # 还原灰度并截取灰度区间
            self.__outImageRGB = numpy.clip(numpy.uint8(out * 255), 0, 255)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 加均匀噪声
    def __addUniformNoise(self):
        if self.__fileName:
            # 起始范围
            low = 100
            # 终止范围
            height = 150
            # 搞一个与图片同规模数组
            out = numpy.zeros(self.__outImageRGB.shape, numpy.uint8)
            # 噪声生成比率
            ratio = 0.05
            # 遍历图片
            for i in range(self.__outImageRGB.shape[0]):
                for j in range(self.__outImageRGB.shape[1]):
                    # 随机数[0.0,1.0)
                    r = random.random()
                    # 填充黑点
                    if r < ratio:
                        # 生成[low，height]的随机值
                        out[i][j] = random.randint(low, height)
                    # 填充白点
                    elif r > 1 - ratio:
                        out[i][j] = random.randint(low, height)
                    # 填充原图
                    else:
                        out[i][j] = self.__outImageRGB[i][j]
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 加脉冲噪声
    def __addImpulseNoise(self):
        if self.__fileName:
            # 搞一个与图片同规模数组
            out = numpy.zeros(self.__outImageRGB.shape, numpy.uint8)
            # 椒盐噪声生成比率
            ratio = 0.05
            # 遍历图片
            for i in range(self.__outImageRGB.shape[0]):
                for j in range(self.__outImageRGB.shape[1]):
                    # 随机数[0.0,1.0)
                    r = random.random()
                    # 填充黑点
                    if r < ratio:
                        out[i][j] = 0
                    # 填充白点
                    elif r > 1 - ratio:
                        out[i][j] = 255
                    # 填充原图
                    else:
                        out[i][j] = self.__outImageRGB[i][j]
            self.__outImageRGB = out.copy()
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # -----------------------------------空域滤波-----------------------------------
    # 均值滤波
    def __meanValueFilter(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.blur(self.__outImageRGB, (5, 5))
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 中值滤波
    def __medianValueFilter(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.medianBlur(self.__outImageRGB, 5)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # Sobel算子锐化
    def __sobel(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.Sobel(self.__outImageRGB, -1, 1, 1, 3)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # Prewitt算子锐化
    def __prewitt(self):
        if self.__fileName:
            # Prewitt 算子
            kernelx = numpy.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
            kernely = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
            # 通过自定义卷积核实现卷积
            imgx = cv2.filter2D(self.__outImageRGB, -1, kernelx)
            imgy = cv2.filter2D(self.__outImageRGB, -1, kernely)
            # 合并
            self.__outImageRGB = cv2.add(imgx, imgy)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # 拉普拉斯算子锐化
    def __laplacian(self):
        if self.__fileName:
            # 直接调库
            self.__outImageRGB = cv2.Laplacian(self.__outImageRGB, -1, ksize=3)
            self.__drawImage(self.outImageView, self.__outImageRGB)

    # -----------------------------------关于-----------------------------------
    # 关于作者
    def __aboutAuthor(self):
        QMessageBox.information(None, '关于作者', '图像处理软件2.1\n\nCopyright © 2021–2099 赵相欣\n\n保留一切权利')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
