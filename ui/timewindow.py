# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'timewindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TimeWindow(object):
    def setupUi(self, TimeWindow):
        TimeWindow.setObjectName("TimeWindow")
        TimeWindow.resize(1224, 862)
        self.label = QtWidgets.QLabel(TimeWindow)
        self.label.setGeometry(QtCore.QRect(20, 10, 72, 15))
        self.label.setObjectName("label")
        self.comboBox = QtWidgets.QComboBox(TimeWindow)
        self.comboBox.setGeometry(QtCore.QRect(20, 60, 331, 21))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.textEdit = QtWidgets.QTextEdit(TimeWindow)
        self.textEdit.setGeometry(QtCore.QRect(20, 160, 441, 561))
        self.textEdit.setObjectName("textEdit")
        self.textEdit_2 = QtWidgets.QTextEdit(TimeWindow)
        self.textEdit_2.setGeometry(QtCore.QRect(710, 160, 481, 551))
        self.textEdit_2.setObjectName("textEdit_2")
        self.label_2 = QtWidgets.QLabel(TimeWindow)
        self.label_2.setGeometry(QtCore.QRect(130, 130, 72, 15))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(TimeWindow)
        self.label_3.setGeometry(QtCore.QRect(920, 120, 72, 20))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(TimeWindow)
        self.label_4.setGeometry(QtCore.QRect(30, 730, 121, 16))
        self.label_4.setObjectName("label_4")
        self.label_windowsmax = QtWidgets.QLabel(TimeWindow)
        self.label_windowsmax.setGeometry(QtCore.QRect(210, 730, 72, 15))
        self.label_windowsmax.setObjectName("label_windowsmax")
        self.label_6 = QtWidgets.QLabel(TimeWindow)
        self.label_6.setGeometry(QtCore.QRect(30, 760, 121, 16))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(TimeWindow)
        self.label_7.setGeometry(QtCore.QRect(30, 790, 121, 16))
        self.label_7.setObjectName("label_7")
        self.label_windowsmin = QtWidgets.QLabel(TimeWindow)
        self.label_windowsmin.setGeometry(QtCore.QRect(210, 760, 72, 15))
        self.label_windowsmin.setObjectName("label_windowsmin")
        self.label_windowssum = QtWidgets.QLabel(TimeWindow)
        self.label_windowssum.setGeometry(QtCore.QRect(210, 790, 72, 15))
        self.label_windowssum.setObjectName("label_windowssum")
        self.label_10 = QtWidgets.QLabel(TimeWindow)
        self.label_10.setGeometry(QtCore.QRect(330, 730, 72, 15))
        self.label_10.setObjectName("label_10")
        self.label_11 = QtWidgets.QLabel(TimeWindow)
        self.label_11.setGeometry(QtCore.QRect(330, 760, 72, 15))
        self.label_11.setObjectName("label_11")
        self.label_windowsmaxnation = QtWidgets.QLabel(TimeWindow)
        self.label_windowsmaxnation.setGeometry(QtCore.QRect(410, 730, 201, 16))
        self.label_windowsmaxnation.setWordWrap(True)
        self.label_windowsmaxnation.setObjectName("label_windowsmaxnation")
        self.label_windowsminnation = QtWidgets.QLabel(TimeWindow)
        self.label_windowsminnation.setGeometry(QtCore.QRect(410, 760, 181, 16))
        self.label_windowsminnation.setWordWrap(True)
        self.label_windowsminnation.setObjectName("label_windowsminnation")
        self.label_5 = QtWidgets.QLabel(TimeWindow)
        self.label_5.setGeometry(QtCore.QRect(710, 740, 161, 16))
        self.label_5.setObjectName("label_5")
        self.label_8 = QtWidgets.QLabel(TimeWindow)
        self.label_8.setGeometry(QtCore.QRect(710, 790, 151, 16))
        self.label_8.setObjectName("label_8")
        self.label_12 = QtWidgets.QLabel(TimeWindow)
        self.label_12.setGeometry(QtCore.QRect(1020, 740, 72, 15))
        self.label_12.setObjectName("label_12")
        self.label_windowsdiffermaxnation = QtWidgets.QLabel(TimeWindow)
        self.label_windowsdiffermaxnation.setGeometry(QtCore.QRect(1060, 740, 161, 16))
        self.label_windowsdiffermaxnation.setWordWrap(True)
        self.label_windowsdiffermaxnation.setObjectName("label_windowsdiffermaxnation")
        self.label_windowsdiffermax = QtWidgets.QLabel(TimeWindow)
        self.label_windowsdiffermax.setGeometry(QtCore.QRect(890, 740, 72, 15))
        self.label_windowsdiffermax.setObjectName("label_windowsdiffermax")
        self.label_windowsdiffersum = QtWidgets.QLabel(TimeWindow)
        self.label_windowsdiffersum.setGeometry(QtCore.QRect(890, 790, 72, 15))
        self.label_windowsdiffersum.setObjectName("label_windowsdiffersum")

        self.retranslateUi(TimeWindow)
        QtCore.QMetaObject.connectSlotsByName(TimeWindow)

    def retranslateUi(self, TimeWindow):
        _translate = QtCore.QCoreApplication.translate
        TimeWindow.setWindowTitle(_translate("TimeWindow", "时间窗口和时间间隙"))
        self.label.setText(_translate("TimeWindow", "所用时间"))
        self.comboBox.setItemText(0, _translate("TimeWindow", "坎普尔 印度"))
        self.comboBox.setItemText(1, _translate("TimeWindow", "堪萨斯城 美国"))
        self.comboBox.setItemText(2, _translate("TimeWindow", "高雄 中国"))
        self.comboBox.setItemText(3, _translate("TimeWindow", "卡拉奇 巴基斯坦"))
        self.comboBox.setItemText(4, _translate("TimeWindow", "加德满都餐厅 尼泊尔"))
        self.comboBox.setItemText(5, _translate("TimeWindow", "考纳斯 立陶宛"))
        self.comboBox.setItemText(6, _translate("TimeWindow", "川崎 日本"))
        self.comboBox.setItemText(7, _translate("TimeWindow", "喀山 俄罗斯"))
        self.comboBox.setItemText(8, _translate("TimeWindow", "喀士穆 苏丹"))
        self.comboBox.setItemText(9, _translate("TimeWindow", "孔敬 泰国"))
        self.comboBox.setItemText(10, _translate("TimeWindow", "库尔纳 孟加拉国"))
        self.comboBox.setItemText(11, _translate("TimeWindow", "基加利 卢旺达"))
        self.comboBox.setItemText(12, _translate("TimeWindow", "京斯敦 澳大利亚"))
        self.label_2.setText(_translate("TimeWindow", "时间窗口"))
        self.label_3.setText(_translate("TimeWindow", "时间间隙"))
        self.label_4.setText(_translate("TimeWindow", "时间窗口最大值为"))
        self.label_windowsmax.setText(_translate("TimeWindow", "TextLabel"))
        self.label_6.setText(_translate("TimeWindow", "时间窗口最小值为"))
        self.label_7.setText(_translate("TimeWindow", "时间窗口累计值为"))
        self.label_windowsmin.setText(_translate("TimeWindow", "TextLabel"))
        self.label_windowssum.setText(_translate("TimeWindow", "TextLabel"))
        self.label_10.setText(_translate("TimeWindow", "国家："))
        self.label_11.setText(_translate("TimeWindow", "国家："))
        self.label_windowsmaxnation.setText(_translate("TimeWindow", "TextLabel"))
        self.label_windowsminnation.setText(_translate("TimeWindow", "TextLabel"))
        self.label_5.setText(_translate("TimeWindow", "时间窗口间隙最大值为"))
        self.label_8.setText(_translate("TimeWindow", "时间窗口间隙累计值为"))
        self.label_12.setText(_translate("TimeWindow", "国家："))
        self.label_windowsdiffermaxnation.setText(_translate("TimeWindow", "TextLabel"))
        self.label_windowsdiffermax.setText(_translate("TimeWindow", "TextLabel"))
        self.label_windowsdiffersum.setText(_translate("TimeWindow", "TextLabel"))