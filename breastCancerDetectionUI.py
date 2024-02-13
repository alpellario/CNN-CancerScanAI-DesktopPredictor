# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\breastCancerDetectionUI.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1024, 680)
        MainWindow.setMinimumSize(QtCore.QSize(1024, 680))
        MainWindow.setMaximumSize(QtCore.QSize(1024, 680))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(0, 0, 1011, 671))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tabStart = QtWidgets.QWidget()
        self.tabStart.setObjectName("tabStart")
        self.label = QtWidgets.QLabel(self.tabStart)
        self.label.setGeometry(QtCore.QRect(200, 220, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.btn_model_upload = QtWidgets.QPushButton(self.tabStart)
        self.btn_model_upload.setGeometry(QtCore.QRect(720, 260, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.btn_model_upload.setFont(font)
        self.btn_model_upload.setObjectName("btn_model_upload")
        self.tb_model_path = QtWidgets.QLineEdit(self.tabStart)
        self.tb_model_path.setGeometry(QtCore.QRect(200, 260, 511, 41))
        self.tb_model_path.setText("")
        self.tb_model_path.setReadOnly(True)
        self.tb_model_path.setObjectName("tb_model_path")
        self.btn_next_1 = QtWidgets.QPushButton(self.tabStart)
        self.btn_next_1.setGeometry(QtCore.QRect(810, 590, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_next_1.setFont(font)
        self.btn_next_1.setObjectName("btn_next_1")
        self.lb_model_success = QtWidgets.QLabel(self.tabStart)
        self.lb_model_success.setGeometry(QtCore.QRect(200, 300, 411, 31))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(39, 124, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(39, 124, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(127, 255, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 255, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 170, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 127, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 255, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        self.lb_model_success.setPalette(palette)
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.lb_model_success.setFont(font)
        self.lb_model_success.setObjectName("lb_model_success")
        self.btn_open_notebook = QtWidgets.QPushButton(self.tabStart)
        self.btn_open_notebook.setGeometry(QtCore.QRect(530, 590, 271, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_open_notebook.setFont(font)
        self.btn_open_notebook.setObjectName("btn_open_notebook")
        self.tabWidget.addTab(self.tabStart, "")
        self.tabDataset = QtWidgets.QWidget()
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(240, 240, 240))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        self.tabDataset.setPalette(palette)
        self.tabDataset.setObjectName("tabDataset")
        self.widget = QtWidgets.QWidget(self.tabDataset)
        self.widget.setGeometry(QtCore.QRect(10, 10, 761, 571))
        self.widget.setMinimumSize(QtCore.QSize(0, 0))
        self.widget.setMaximumSize(QtCore.QSize(791, 611))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 145, 234))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 127))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 145, 234))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 145, 234))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 145, 234))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        self.widget.setPalette(palette)
        self.widget.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.widget.setObjectName("widget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.widget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSizeConstraint(QtWidgets.QLayout.SetMaximumSize)
        self.gridLayout.setHorizontalSpacing(42)
        self.gridLayout.setVerticalSpacing(35)
        self.gridLayout.setObjectName("gridLayout")
        self.p14 = QtWidgets.QLabel(self.widget)
        self.p14.setMaximumSize(QtCore.QSize(100, 100))
        self.p14.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p14.setLineWidth(1)
        self.p14.setAlignment(QtCore.Qt.AlignCenter)
        self.p14.setObjectName("p14")
        self.gridLayout.addWidget(self.p14, 2, 3, 1, 1)
        self.p11 = QtWidgets.QLabel(self.widget)
        self.p11.setMaximumSize(QtCore.QSize(100, 100))
        self.p11.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p11.setLineWidth(1)
        self.p11.setAlignment(QtCore.Qt.AlignCenter)
        self.p11.setObjectName("p11")
        self.gridLayout.addWidget(self.p11, 2, 0, 1, 1)
        self.p8 = QtWidgets.QLabel(self.widget)
        self.p8.setMaximumSize(QtCore.QSize(100, 100))
        self.p8.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p8.setLineWidth(1)
        self.p8.setAlignment(QtCore.Qt.AlignCenter)
        self.p8.setObjectName("p8")
        self.gridLayout.addWidget(self.p8, 1, 2, 1, 1)
        self.p3 = QtWidgets.QLabel(self.widget)
        self.p3.setMaximumSize(QtCore.QSize(100, 100))
        self.p3.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p3.setLineWidth(1)
        self.p3.setAlignment(QtCore.Qt.AlignCenter)
        self.p3.setObjectName("p3")
        self.gridLayout.addWidget(self.p3, 0, 2, 1, 1)
        self.p20 = QtWidgets.QLabel(self.widget)
        self.p20.setMaximumSize(QtCore.QSize(100, 100))
        self.p20.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p20.setLineWidth(1)
        self.p20.setAlignment(QtCore.Qt.AlignCenter)
        self.p20.setObjectName("p20")
        self.gridLayout.addWidget(self.p20, 3, 4, 1, 1)
        self.p18 = QtWidgets.QLabel(self.widget)
        self.p18.setMaximumSize(QtCore.QSize(100, 100))
        self.p18.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p18.setLineWidth(1)
        self.p18.setAlignment(QtCore.Qt.AlignCenter)
        self.p18.setObjectName("p18")
        self.gridLayout.addWidget(self.p18, 3, 2, 1, 1)
        self.p19 = QtWidgets.QLabel(self.widget)
        self.p19.setMaximumSize(QtCore.QSize(100, 100))
        self.p19.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p19.setLineWidth(1)
        self.p19.setAlignment(QtCore.Qt.AlignCenter)
        self.p19.setObjectName("p19")
        self.gridLayout.addWidget(self.p19, 3, 3, 1, 1)
        self.p12 = QtWidgets.QLabel(self.widget)
        self.p12.setMaximumSize(QtCore.QSize(100, 100))
        self.p12.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p12.setLineWidth(1)
        self.p12.setAlignment(QtCore.Qt.AlignCenter)
        self.p12.setObjectName("p12")
        self.gridLayout.addWidget(self.p12, 2, 1, 1, 1)
        self.p13 = QtWidgets.QLabel(self.widget)
        self.p13.setMaximumSize(QtCore.QSize(100, 100))
        self.p13.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p13.setLineWidth(1)
        self.p13.setAlignment(QtCore.Qt.AlignCenter)
        self.p13.setObjectName("p13")
        self.gridLayout.addWidget(self.p13, 2, 2, 1, 1)
        self.p6 = QtWidgets.QLabel(self.widget)
        self.p6.setMaximumSize(QtCore.QSize(100, 100))
        self.p6.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p6.setLineWidth(1)
        self.p6.setAlignment(QtCore.Qt.AlignCenter)
        self.p6.setObjectName("p6")
        self.gridLayout.addWidget(self.p6, 1, 0, 1, 1)
        self.p1 = QtWidgets.QLabel(self.widget)
        self.p1.setMaximumSize(QtCore.QSize(100, 100))
        self.p1.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p1.setLineWidth(1)
        self.p1.setAlignment(QtCore.Qt.AlignCenter)
        self.p1.setObjectName("p1")
        self.gridLayout.addWidget(self.p1, 0, 0, 1, 1)
        self.p2 = QtWidgets.QLabel(self.widget)
        self.p2.setMaximumSize(QtCore.QSize(100, 100))
        self.p2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p2.setLineWidth(1)
        self.p2.setAlignment(QtCore.Qt.AlignCenter)
        self.p2.setObjectName("p2")
        self.gridLayout.addWidget(self.p2, 0, 1, 1, 1)
        self.p17 = QtWidgets.QLabel(self.widget)
        self.p17.setMaximumSize(QtCore.QSize(100, 100))
        self.p17.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p17.setLineWidth(1)
        self.p17.setAlignment(QtCore.Qt.AlignCenter)
        self.p17.setObjectName("p17")
        self.gridLayout.addWidget(self.p17, 3, 1, 1, 1)
        self.p16 = QtWidgets.QLabel(self.widget)
        self.p16.setMaximumSize(QtCore.QSize(100, 100))
        self.p16.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p16.setLineWidth(1)
        self.p16.setAlignment(QtCore.Qt.AlignCenter)
        self.p16.setObjectName("p16")
        self.gridLayout.addWidget(self.p16, 3, 0, 1, 1)
        self.p7 = QtWidgets.QLabel(self.widget)
        self.p7.setMaximumSize(QtCore.QSize(100, 100))
        self.p7.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p7.setLineWidth(1)
        self.p7.setAlignment(QtCore.Qt.AlignCenter)
        self.p7.setObjectName("p7")
        self.gridLayout.addWidget(self.p7, 1, 1, 1, 1)
        self.p15 = QtWidgets.QLabel(self.widget)
        self.p15.setMaximumSize(QtCore.QSize(100, 100))
        self.p15.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p15.setLineWidth(1)
        self.p15.setAlignment(QtCore.Qt.AlignCenter)
        self.p15.setObjectName("p15")
        self.gridLayout.addWidget(self.p15, 2, 4, 1, 1)
        self.p5 = QtWidgets.QLabel(self.widget)
        self.p5.setMaximumSize(QtCore.QSize(100, 100))
        self.p5.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p5.setLineWidth(1)
        self.p5.setAlignment(QtCore.Qt.AlignCenter)
        self.p5.setObjectName("p5")
        self.gridLayout.addWidget(self.p5, 0, 4, 1, 1)
        self.p10 = QtWidgets.QLabel(self.widget)
        self.p10.setMaximumSize(QtCore.QSize(100, 100))
        self.p10.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p10.setLineWidth(1)
        self.p10.setAlignment(QtCore.Qt.AlignCenter)
        self.p10.setObjectName("p10")
        self.gridLayout.addWidget(self.p10, 1, 4, 1, 1)
        self.p4 = QtWidgets.QLabel(self.widget)
        self.p4.setMaximumSize(QtCore.QSize(100, 100))
        self.p4.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p4.setLineWidth(1)
        self.p4.setAlignment(QtCore.Qt.AlignCenter)
        self.p4.setObjectName("p4")
        self.gridLayout.addWidget(self.p4, 0, 3, 1, 1)
        self.p9 = QtWidgets.QLabel(self.widget)
        self.p9.setMaximumSize(QtCore.QSize(100, 100))
        self.p9.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.p9.setLineWidth(1)
        self.p9.setAlignment(QtCore.Qt.AlignCenter)
        self.p9.setObjectName("p9")
        self.gridLayout.addWidget(self.p9, 1, 3, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 1, 1, 1)
        self.btn_gallery_back = QtWidgets.QPushButton(self.tabDataset)
        self.btn_gallery_back.setGeometry(QtCore.QRect(260, 590, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_gallery_back.setFont(font)
        self.btn_gallery_back.setObjectName("btn_gallery_back")
        self.btn_gallery_next = QtWidgets.QPushButton(self.tabDataset)
        self.btn_gallery_next.setGeometry(QtCore.QRect(420, 590, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.btn_gallery_next.setFont(font)
        self.btn_gallery_next.setObjectName("btn_gallery_next")
        self.lb_current_page = QtWidgets.QLabel(self.tabDataset)
        self.lb_current_page.setGeometry(QtCore.QRect(390, 590, 31, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lb_current_page.setFont(font)
        self.lb_current_page.setObjectName("lb_current_page")
        self.label_2 = QtWidgets.QLabel(self.tabDataset)
        self.label_2.setGeometry(QtCore.QRect(780, 20, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.lb_img_name = QtWidgets.QLabel(self.tabDataset)
        self.lb_img_name.setGeometry(QtCore.QRect(780, 50, 231, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(15)
        font.setBold(False)
        font.setWeight(50)
        self.lb_img_name.setFont(font)
        self.lb_img_name.setText("")
        self.lb_img_name.setObjectName("lb_img_name")
        self.label_16 = QtWidgets.QLabel(self.tabDataset)
        self.label_16.setGeometry(QtCore.QRect(20, 590, 111, 31))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.lb_total_page = QtWidgets.QLabel(self.tabDataset)
        self.lb_total_page.setGeometry(QtCore.QRect(140, 590, 71, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.lb_total_page.setFont(font)
        self.lb_total_page.setObjectName("lb_total_page")
        self.btn_switch_test = QtWidgets.QPushButton(self.tabDataset)
        self.btn_switch_test.setGeometry(QtCore.QRect(780, 530, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.btn_switch_test.setFont(font)
        self.btn_switch_test.setObjectName("btn_switch_test")
        self.btn_switch_train = QtWidgets.QPushButton(self.tabDataset)
        self.btn_switch_train.setGeometry(QtCore.QRect(780, 480, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.btn_switch_train.setFont(font)
        self.btn_switch_train.setObjectName("btn_switch_train")
        self.lb_img_label = QtWidgets.QLabel(self.tabDataset)
        self.lb_img_label.setGeometry(QtCore.QRect(10, 10, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(20)
        font.setBold(False)
        font.setWeight(50)
        self.lb_img_label.setFont(font)
        self.lb_img_label.setObjectName("lb_img_label")
        self.btn_img_predict = QtWidgets.QPushButton(self.tabDataset)
        self.btn_img_predict.setGeometry(QtCore.QRect(780, 100, 211, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.btn_img_predict.setFont(font)
        self.btn_img_predict.setObjectName("btn_img_predict")
        self.btn_random_predict = QtWidgets.QPushButton(self.tabDataset)
        self.btn_random_predict.setGeometry(QtCore.QRect(780, 160, 211, 91))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        self.btn_random_predict.setFont(font)
        self.btn_random_predict.setObjectName("btn_random_predict")
        self.tabWidget.addTab(self.tabDataset, "")
        self.tabPrediction = QtWidgets.QWidget()
        self.tabPrediction.setObjectName("tabPrediction")
        self.groupBox = QtWidgets.QGroupBox(self.tabPrediction)
        self.groupBox.setGeometry(QtCore.QRect(180, 20, 611, 511))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        self.groupBox.setFont(font)
        self.groupBox.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.groupBox.setObjectName("groupBox")
        self.p_image = QtWidgets.QLabel(self.groupBox)
        self.p_image.setGeometry(QtCore.QRect(190, 150, 260, 260))
        self.p_image.setStyleSheet("background-color: white;")
        self.p_image.setAlignment(QtCore.Qt.AlignCenter)
        self.p_image.setObjectName("p_image")
        self.p_border = QtWidgets.QLabel(self.groupBox)
        self.p_border.setGeometry(QtCore.QRect(170, 130, 300, 300))
        self.p_border.setStyleSheet("background-color: red;")
        self.p_border.setText("")
        self.p_border.setObjectName("p_border")
        self.lb_true = QtWidgets.QLabel(self.groupBox)
        self.lb_true.setGeometry(QtCore.QRect(140, 30, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.lb_true.setFont(font)
        self.lb_true.setObjectName("lb_true")
        self.label_5 = QtWidgets.QLabel(self.groupBox)
        self.label_5.setGeometry(QtCore.QRect(60, 30, 71, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.lb_predict = QtWidgets.QLabel(self.groupBox)
        self.lb_predict.setGeometry(QtCore.QRect(140, 70, 281, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        font.setBold(False)
        font.setWeight(50)
        self.lb_predict.setFont(font)
        self.lb_predict.setObjectName("lb_predict")
        self.label_6 = QtWidgets.QLabel(self.groupBox)
        self.label_6.setGeometry(QtCore.QRect(20, 70, 111, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.groupBox)
        self.label_7.setGeometry(QtCore.QRect(20, 450, 121, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setObjectName("label_7")
        self.lb_confidence = QtWidgets.QLabel(self.groupBox)
        self.lb_confidence.setGeometry(QtCore.QRect(150, 450, 141, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(14)
        font.setBold(False)
        font.setWeight(50)
        self.lb_confidence.setFont(font)
        self.lb_confidence.setObjectName("lb_confidence")
        self.p_border.raise_()
        self.p_image.raise_()
        self.lb_true.raise_()
        self.label_5.raise_()
        self.lb_predict.raise_()
        self.label_6.raise_()
        self.label_7.raise_()
        self.lb_confidence.raise_()
        self.btn_new_image = QtWidgets.QPushButton(self.tabPrediction)
        self.btn_new_image.setGeometry(QtCore.QRect(410, 550, 191, 41))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(12)
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        self.btn_new_image.setFont(font)
        self.btn_new_image.setObjectName("btn_new_image")
        self.tabWidget.addTab(self.tabPrediction, "")
        self.tabMetrics = QtWidgets.QWidget()
        self.tabMetrics.setObjectName("tabMetrics")
        self.cb_graphics = QtWidgets.QComboBox(self.tabMetrics)
        self.cb_graphics.setGeometry(QtCore.QRect(30, 60, 241, 41))
        self.cb_graphics.setObjectName("cb_graphics")
        self.cb_graphics.addItem("")
        self.cb_graphics.addItem("")
        self.cb_graphics.addItem("")
        self.label_9 = QtWidgets.QLabel(self.tabMetrics)
        self.label_9.setGeometry(QtCore.QRect(30, 20, 201, 31))
        font = QtGui.QFont()
        font.setFamily("Century Gothic")
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setObjectName("label_9")
        self.p_graph = QtWidgets.QLabel(self.tabMetrics)
        self.p_graph.setGeometry(QtCore.QRect(360, 20, 600, 600))
        self.p_graph.setStyleSheet("border: 1px solid black;")
        self.p_graph.setText("")
        self.p_graph.setObjectName("p_graph")
        self.tabWidget.addTab(self.tabMetrics, "")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Breast Cancer Detection Project"))
        self.label.setText(_translate("MainWindow", "Model Weights Path:"))
        self.btn_model_upload.setText(_translate("MainWindow", "UPLOAD"))
        self.btn_next_1.setText(_translate("MainWindow", "CONTINUE =>"))
        self.lb_model_success.setText(_translate("MainWindow", "Model weights loaded."))
        self.btn_open_notebook.setText(_translate("MainWindow", "Open Training Notebook File"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabStart), _translate("MainWindow", "Initial Settings"))
        self.p14.setText(_translate("MainWindow", "TextLabel"))
        self.p11.setText(_translate("MainWindow", "TextLabel"))
        self.p8.setText(_translate("MainWindow", "TextLabel"))
        self.p3.setText(_translate("MainWindow", "TextLabel"))
        self.p20.setText(_translate("MainWindow", "TextLabel"))
        self.p18.setText(_translate("MainWindow", "TextLabel"))
        self.p19.setText(_translate("MainWindow", "TextLabel"))
        self.p12.setText(_translate("MainWindow", "TextLabel"))
        self.p13.setText(_translate("MainWindow", "TextLabel"))
        self.p6.setText(_translate("MainWindow", "TextLabel"))
        self.p1.setText(_translate("MainWindow", "TextLabel"))
        self.p2.setText(_translate("MainWindow", "TextLabel"))
        self.p17.setText(_translate("MainWindow", "TextLabel"))
        self.p16.setText(_translate("MainWindow", "TextLabel"))
        self.p7.setText(_translate("MainWindow", "TextLabel"))
        self.p15.setText(_translate("MainWindow", "TextLabel"))
        self.p5.setText(_translate("MainWindow", "TextLabel"))
        self.p10.setText(_translate("MainWindow", "TextLabel"))
        self.p4.setText(_translate("MainWindow", "TextLabel"))
        self.p9.setText(_translate("MainWindow", "TextLabel"))
        self.btn_gallery_back.setText(_translate("MainWindow", "<"))
        self.btn_gallery_next.setText(_translate("MainWindow", ">"))
        self.lb_current_page.setText(_translate("MainWindow", "1"))
        self.label_2.setText(_translate("MainWindow", "Image Name:"))
        self.label_16.setText(_translate("MainWindow", "Total Pages:"))
        self.lb_total_page.setText(_translate("MainWindow", "525"))
        self.btn_switch_test.setText(_translate("MainWindow", "Switch to Test Dataset"))
        self.btn_switch_train.setText(_translate("MainWindow", "Switch to Train Dataset"))
        self.lb_img_label.setText(_translate("MainWindow", "No selection made."))
        self.btn_img_predict.setText(_translate("MainWindow", "MAKE PREDICTION"))
        self.btn_random_predict.setText(_translate("MainWindow", "Random 36 Images\n"
"from Test Dataset"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabDataset), _translate("MainWindow", "Dataset"))
        self.groupBox.setTitle(_translate("MainWindow", "Prediction Result"))
        self.p_image.setText(_translate("MainWindow", "IMAGE"))
        self.lb_true.setText(_translate("MainWindow", "Cancerous Cell"))
        self.label_5.setText(_translate("MainWindow", "Actual:"))
        self.lb_predict.setText(_translate("MainWindow", "Cancerous Cell"))
        self.label_6.setText(_translate("MainWindow", "Prediction:"))
        self.label_7.setText(_translate("MainWindow", "Confidence :"))
        self.lb_confidence.setText(_translate("MainWindow", "%56"))
        self.btn_new_image.setText(_translate("MainWindow", "New Image Selection"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabPrediction), _translate("MainWindow", "Prediction"))
        self.cb_graphics.setItemText(0, _translate("MainWindow", "Model Loss"))
        self.cb_graphics.setItemText(1, _translate("MainWindow", "Model Accuracy"))
        self.cb_graphics.setItemText(2, _translate("MainWindow", "Confusion Matrix"))
        self.label_9.setText(_translate("MainWindow", "Charts"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabMetrics), _translate("MainWindow", "Metrics and Charts"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
