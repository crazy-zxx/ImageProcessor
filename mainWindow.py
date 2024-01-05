# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(896, 577)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.outImageView = QtWidgets.QGraphicsView(self.centralwidget)
        self.outImageView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.outImageView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.outImageView.setObjectName("outImageView")
        self.gridLayout.addWidget(self.outImageView, 1, 1, 1, 1)
        self.srcImageView = QtWidgets.QGraphicsView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.srcImageView.sizePolicy().hasHeightForWidth())
        self.srcImageView.setSizePolicy(sizePolicy)
        self.srcImageView.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.srcImageView.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.srcImageView.setObjectName("srcImageView")
        self.gridLayout.addWidget(self.srcImageView, 1, 0, 1, 1)
        self.srcImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.srcImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.srcImageLabel.setObjectName("srcImageLabel")
        self.gridLayout.addWidget(self.srcImageLabel, 0, 0, 1, 1)
        self.outImageLabel = QtWidgets.QLabel(self.centralwidget)
        self.outImageLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.outImageLabel.setObjectName("outImageLabel")
        self.gridLayout.addWidget(self.outImageLabel, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 896, 24))
        self.menubar.setObjectName("menubar")
        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setObjectName("fileMenu")
        self.resetImageMenu = QtWidgets.QMenu(self.menubar)
        self.resetImageMenu.setObjectName("resetImageMenu")
        self.aboutMenu = QtWidgets.QMenu(self.menubar)
        self.aboutMenu.setObjectName("aboutMenu")
        self.grayMappingMenu = QtWidgets.QMenu(self.menubar)
        self.grayMappingMenu.setObjectName("grayMappingMenu")
        self.operateImageMenu = QtWidgets.QMenu(self.menubar)
        self.operateImageMenu.setObjectName("operateImageMenu")
        self.histogramMenu = QtWidgets.QMenu(self.menubar)
        self.histogramMenu.setObjectName("histogramMenu")
        self.noiseMenu = QtWidgets.QMenu(self.menubar)
        self.noiseMenu.setObjectName("noiseMenu")
        self.filterMenu = QtWidgets.QMenu(self.menubar)
        self.filterMenu.setObjectName("filterMenu")
        self.smoothMenu = QtWidgets.QMenu(self.filterMenu)
        self.smoothMenu.setObjectName("smoothMenu")
        self.sharpMenu = QtWidgets.QMenu(self.filterMenu)
        self.sharpMenu.setObjectName("sharpMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.openFileAction = QtWidgets.QAction(MainWindow)
        self.openFileAction.setObjectName("openFileAction")
        self.saveFileAction = QtWidgets.QAction(MainWindow)
        self.saveFileAction.setObjectName("saveFileAction")
        self.saveFileAsAction = QtWidgets.QAction(MainWindow)
        self.saveFileAsAction.setObjectName("saveFileAsAction")
        self.exitAppAction = QtWidgets.QAction(MainWindow)
        self.exitAppAction.setObjectName("exitAppAction")
        self.resetImageAction = QtWidgets.QAction(MainWindow)
        self.resetImageAction.setObjectName("resetImageAction")
        self.aboutAction = QtWidgets.QAction(MainWindow)
        self.aboutAction.setObjectName("aboutAction")
        self.actiongg_2 = QtWidgets.QAction(MainWindow)
        self.actiongg_2.setObjectName("actiongg_2")
        self.grayAction = QtWidgets.QAction(MainWindow)
        self.grayAction.setObjectName("grayAction")
        self.binaryAction = QtWidgets.QAction(MainWindow)
        self.binaryAction.setObjectName("binaryAction")
        self.reverseAction = QtWidgets.QAction(MainWindow)
        self.reverseAction.setObjectName("reverseAction")
        self.imageAddAction = QtWidgets.QAction(MainWindow)
        self.imageAddAction.setObjectName("imageAddAction")
        self.imageSubtractAction = QtWidgets.QAction(MainWindow)
        self.imageSubtractAction.setObjectName("imageSubtractAction")
        self.imageMultiplyAction = QtWidgets.QAction(MainWindow)
        self.imageMultiplyAction.setObjectName("imageMultiplyAction")
        self.histogramAction = QtWidgets.QAction(MainWindow)
        self.histogramAction.setObjectName("histogramAction")
        self.histogramEqAction = QtWidgets.QAction(MainWindow)
        self.histogramEqAction.setObjectName("histogramEqAction")
        self.lightAction = QtWidgets.QAction(MainWindow)
        self.lightAction.setObjectName("lightAction")
        self.contrastAction = QtWidgets.QAction(MainWindow)
        self.contrastAction.setObjectName("contrastAction")
        self.sharpAction = QtWidgets.QAction(MainWindow)
        self.sharpAction.setObjectName("sharpAction")
        self.zoomAction = QtWidgets.QAction(MainWindow)
        self.zoomAction.setObjectName("zoomAction")
        self.rotateAction = QtWidgets.QAction(MainWindow)
        self.rotateAction.setObjectName("rotateAction")
        self.actiongg = QtWidgets.QAction(MainWindow)
        self.actiongg.setObjectName("actiongg")
        self.saturationAction = QtWidgets.QAction(MainWindow)
        self.saturationAction.setObjectName("saturationAction")
        self.hueAction = QtWidgets.QAction(MainWindow)
        self.hueAction.setObjectName("hueAction")
        self.reColorAction = QtWidgets.QAction(MainWindow)
        self.reColorAction.setObjectName("reColorAction")
        self.addGaussianNoiseAction = QtWidgets.QAction(MainWindow)
        self.addGaussianNoiseAction.setObjectName("addGaussianNoiseAction")
        self.actiongg_3 = QtWidgets.QAction(MainWindow)
        self.actiongg_3.setObjectName("actiongg_3")
        self.actiongg_4 = QtWidgets.QAction(MainWindow)
        self.actiongg_4.setObjectName("actiongg_4")
        self.meanValueAction = QtWidgets.QAction(MainWindow)
        self.meanValueAction.setObjectName("meanValueAction")
        self.medianValueAction = QtWidgets.QAction(MainWindow)
        self.medianValueAction.setObjectName("medianValueAction")
        self.sobelAction = QtWidgets.QAction(MainWindow)
        self.sobelAction.setObjectName("sobelAction")
        self.prewittAction = QtWidgets.QAction(MainWindow)
        self.prewittAction.setObjectName("prewittAction")
        self.laplacianAction = QtWidgets.QAction(MainWindow)
        self.laplacianAction.setObjectName("laplacianAction")
        self.addUiformNoiseAction = QtWidgets.QAction(MainWindow)
        self.addUiformNoiseAction.setObjectName("addUiformNoiseAction")
        self.addImpulseNoiseAction = QtWidgets.QAction(MainWindow)
        self.addImpulseNoiseAction.setObjectName("addImpulseNoiseAction")
        self.fileMenu.addAction(self.openFileAction)
        self.fileMenu.addAction(self.saveFileAction)
        self.fileMenu.addAction(self.saveFileAsAction)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAppAction)
        self.resetImageMenu.addAction(self.resetImageAction)
        self.aboutMenu.addAction(self.aboutAction)
        self.grayMappingMenu.addAction(self.grayAction)
        self.grayMappingMenu.addAction(self.binaryAction)
        self.grayMappingMenu.addAction(self.reverseAction)
        self.grayMappingMenu.addSeparator()
        self.grayMappingMenu.addAction(self.lightAction)
        self.grayMappingMenu.addAction(self.contrastAction)
        self.grayMappingMenu.addAction(self.sharpAction)
        self.grayMappingMenu.addAction(self.saturationAction)
        self.grayMappingMenu.addAction(self.hueAction)
        self.operateImageMenu.addAction(self.imageAddAction)
        self.operateImageMenu.addAction(self.imageSubtractAction)
        self.operateImageMenu.addAction(self.imageMultiplyAction)
        self.operateImageMenu.addSeparator()
        self.operateImageMenu.addAction(self.zoomAction)
        self.operateImageMenu.addAction(self.rotateAction)
        self.histogramMenu.addAction(self.histogramAction)
        self.histogramMenu.addAction(self.histogramEqAction)
        self.noiseMenu.addAction(self.addGaussianNoiseAction)
        self.noiseMenu.addAction(self.addUiformNoiseAction)
        self.noiseMenu.addAction(self.addImpulseNoiseAction)
        self.smoothMenu.addAction(self.meanValueAction)
        self.smoothMenu.addAction(self.medianValueAction)
        self.sharpMenu.addAction(self.sobelAction)
        self.sharpMenu.addAction(self.prewittAction)
        self.sharpMenu.addAction(self.laplacianAction)
        self.filterMenu.addAction(self.smoothMenu.menuAction())
        self.filterMenu.addAction(self.sharpMenu.menuAction())
        self.menubar.addAction(self.fileMenu.menuAction())
        self.menubar.addAction(self.resetImageMenu.menuAction())
        self.menubar.addAction(self.grayMappingMenu.menuAction())
        self.menubar.addAction(self.operateImageMenu.menuAction())
        self.menubar.addAction(self.histogramMenu.menuAction())
        self.menubar.addAction(self.noiseMenu.menuAction())
        self.menubar.addAction(self.filterMenu.menuAction())
        self.menubar.addAction(self.aboutMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像处理软件"))
        self.srcImageLabel.setText(_translate("MainWindow", "原图预览"))
        self.outImageLabel.setText(_translate("MainWindow", "处理后的图片预览"))
        self.fileMenu.setTitle(_translate("MainWindow", "文件"))
        self.resetImageMenu.setTitle(_translate("MainWindow", "重置图片"))
        self.aboutMenu.setTitle(_translate("MainWindow", "关于"))
        self.grayMappingMenu.setTitle(_translate("MainWindow", "直接灰度映射"))
        self.operateImageMenu.setTitle(_translate("MainWindow", "图像运算"))
        self.histogramMenu.setTitle(_translate("MainWindow", "直方图均衡"))
        self.noiseMenu.setTitle(_translate("MainWindow", "噪声"))
        self.filterMenu.setTitle(_translate("MainWindow", "空域滤波"))
        self.smoothMenu.setTitle(_translate("MainWindow", "平滑滤波器"))
        self.sharpMenu.setTitle(_translate("MainWindow", "锐化滤波器"))
        self.openFileAction.setText(_translate("MainWindow", "打开"))
        self.saveFileAction.setText(_translate("MainWindow", "保存"))
        self.saveFileAsAction.setText(_translate("MainWindow", "另存为"))
        self.exitAppAction.setText(_translate("MainWindow", "退出"))
        self.resetImageAction.setText(_translate("MainWindow", "恢复到原始图片"))
        self.aboutAction.setText(_translate("MainWindow", "关于作者"))
        self.actiongg_2.setText(_translate("MainWindow", "gg"))
        self.grayAction.setText(_translate("MainWindow", "灰度化"))
        self.binaryAction.setText(_translate("MainWindow", "二值化"))
        self.reverseAction.setText(_translate("MainWindow", "颜色反转"))
        self.imageAddAction.setText(_translate("MainWindow", "加"))
        self.imageSubtractAction.setText(_translate("MainWindow", "减"))
        self.imageMultiplyAction.setText(_translate("MainWindow", "乘"))
        self.histogramAction.setText(_translate("MainWindow", "归一化直方图"))
        self.histogramEqAction.setText(_translate("MainWindow", "直方图均衡化"))
        self.lightAction.setText(_translate("MainWindow", "亮度"))
        self.contrastAction.setText(_translate("MainWindow", "对比度"))
        self.sharpAction.setText(_translate("MainWindow", "锐度"))
        self.zoomAction.setText(_translate("MainWindow", "缩放"))
        self.rotateAction.setText(_translate("MainWindow", "旋转"))
        self.actiongg.setText(_translate("MainWindow", "gg"))
        self.saturationAction.setText(_translate("MainWindow", "饱和度"))
        self.hueAction.setText(_translate("MainWindow", "色调"))
        self.reColorAction.setText(_translate("MainWindow", "重新着色"))
        self.addGaussianNoiseAction.setText(_translate("MainWindow", "加高斯噪声"))
        self.actiongg_3.setText(_translate("MainWindow", "gg"))
        self.actiongg_4.setText(_translate("MainWindow", "gg"))
        self.meanValueAction.setText(_translate("MainWindow", "均值滤波器"))
        self.medianValueAction.setText(_translate("MainWindow", "中值滤波器"))
        self.sobelAction.setText(_translate("MainWindow", "Sobel算子"))
        self.prewittAction.setText(_translate("MainWindow", "Prewitt算子"))
        self.laplacianAction.setText(_translate("MainWindow", "拉普拉斯算子"))
        self.addUiformNoiseAction.setText(_translate("MainWindow", "加均匀噪声"))
        self.addImpulseNoiseAction.setText(_translate("MainWindow", "加脉冲噪声"))
