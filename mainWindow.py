# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(696, 524)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.outImageView = QtWidgets.QGraphicsView(self.centralwidget)
        self.outImageView.setObjectName("outImageView")
        self.gridLayout.addWidget(self.outImageView, 1, 1, 1, 1)
        self.srcImageView = QtWidgets.QGraphicsView(self.centralwidget)
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
        self.menubar.setGeometry(QtCore.QRect(0, 0, 696, 24))
        self.menubar.setObjectName("menubar")
        self.fileMenu = QtWidgets.QMenu(self.menubar)
        self.fileMenu.setObjectName("fileMenu")
        self.resetImageMenu = QtWidgets.QMenu(self.menubar)
        self.resetImageMenu.setObjectName("resetImageMenu")
        self.aboutMenu = QtWidgets.QMenu(self.menubar)
        self.aboutMenu.setObjectName("aboutMenu")
        self.preImageMenu = QtWidgets.QMenu(self.menubar)
        self.preImageMenu.setObjectName("preImageMenu")
        self.operateImageMenu = QtWidgets.QMenu(self.menubar)
        self.operateImageMenu.setObjectName("operateImageMenu")
        self.histogramMenu = QtWidgets.QMenu(self.menubar)
        self.histogramMenu.setObjectName("histogramMenu")
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
        self.addAction = QtWidgets.QAction(MainWindow)
        self.addAction.setObjectName("addAction")
        self.subtractAction = QtWidgets.QAction(MainWindow)
        self.subtractAction.setObjectName("subtractAction")
        self.multiplyAction = QtWidgets.QAction(MainWindow)
        self.multiplyAction.setObjectName("multiplyAction")
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
        self.fileMenu.addAction(self.openFileAction)
        self.fileMenu.addAction(self.saveFileAction)
        self.fileMenu.addAction(self.saveFileAsAction)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAppAction)
        self.resetImageMenu.addAction(self.resetImageAction)
        self.aboutMenu.addAction(self.aboutAction)
        self.preImageMenu.addAction(self.grayAction)
        self.preImageMenu.addAction(self.binaryAction)
        self.preImageMenu.addAction(self.reverseAction)
        self.preImageMenu.addSeparator()
        self.preImageMenu.addAction(self.lightAction)
        self.preImageMenu.addAction(self.contrastAction)
        self.preImageMenu.addAction(self.sharpAction)
        self.preImageMenu.addAction(self.saturationAction)
        self.operateImageMenu.addAction(self.addAction)
        self.operateImageMenu.addAction(self.subtractAction)
        self.operateImageMenu.addAction(self.multiplyAction)
        self.operateImageMenu.addSeparator()
        self.operateImageMenu.addAction(self.zoomAction)
        self.operateImageMenu.addAction(self.rotateAction)
        self.histogramMenu.addAction(self.histogramAction)
        self.histogramMenu.addAction(self.histogramEqAction)
        self.menubar.addAction(self.fileMenu.menuAction())
        self.menubar.addAction(self.resetImageMenu.menuAction())
        self.menubar.addAction(self.preImageMenu.menuAction())
        self.menubar.addAction(self.operateImageMenu.menuAction())
        self.menubar.addAction(self.histogramMenu.menuAction())
        self.menubar.addAction(self.aboutMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "图像处理软件2.0"))
        self.srcImageLabel.setText(_translate("MainWindow", "原图预览"))
        self.outImageLabel.setText(_translate("MainWindow", "处理后的图片预览"))
        self.fileMenu.setTitle(_translate("MainWindow", "文件"))
        self.resetImageMenu.setTitle(_translate("MainWindow", "重置图片"))
        self.aboutMenu.setTitle(_translate("MainWindow", "关于"))
        self.preImageMenu.setTitle(_translate("MainWindow", "图像预处理"))
        self.operateImageMenu.setTitle(_translate("MainWindow", "图像运算"))
        self.histogramMenu.setTitle(_translate("MainWindow", "直方图均衡"))
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
        self.addAction.setText(_translate("MainWindow", "加"))
        self.subtractAction.setText(_translate("MainWindow", "减"))
        self.multiplyAction.setText(_translate("MainWindow", "乘"))
        self.histogramAction.setText(_translate("MainWindow", "归一化直方图"))
        self.histogramEqAction.setText(_translate("MainWindow", "直方图均衡化"))
        self.lightAction.setText(_translate("MainWindow", "亮度"))
        self.contrastAction.setText(_translate("MainWindow", "对比度"))
        self.sharpAction.setText(_translate("MainWindow", "锐度"))
        self.zoomAction.setText(_translate("MainWindow", "缩放"))
        self.rotateAction.setText(_translate("MainWindow", "旋转"))
        self.actiongg.setText(_translate("MainWindow", "gg"))
        self.saturationAction.setText(_translate("MainWindow", "饱和度"))
