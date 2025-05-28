# -*- coding: utf-8 -*-
import random
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget
import pyqtgraph as pg, numpy as np, cv2


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(899, 628)

        self.verticalLayout_4 = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout_4.setContentsMargins(9, 9, 9, 9)
        self.verticalLayout_4.setSpacing(9)
        self.verticalLayout_4.setObjectName("verticalLayout_4")

        self.grafView = PlotWidget()
        self.grafView.showGrid(x=True, y=True)
        self.grafView.setLabels(left="Percentage", bottom="Frame")
        self.grafView.setObjectName("grafView")

        # Initialize data lists
        self.data = []
        self.data1 = []

        # Initialize plot curves
        self.curve1 = self.grafView.plot(pen=pg.mkPen(color=(200, 1, 1), width=2))
        self.curve2 = self.grafView.plot(pen=pg.mkPen(color=(1, 1, 255), width=2))

        # Set axis limits
        self.grafView.setXRange(0, 100)
        self.grafView.setYRange(0, 10)

        self.loadVid()

        self.verticalLayout_4.addWidget(self.grafView)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def loadVid(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.updateFrame)
        self.timer.start(30)  # Update at ~30fps

    def updateFrame(self):
        if True:
            # Generate new random data
            if len(self.data) >= 100:  # Limit to 100 data points
                self.data = self.data[1:]
                self.data1 = self.data1[1:]

            r1 = random.randint(1, 10)
            r2 = random.randint(1, 10)
            self.data.append(r1)
            self.data1.append(r2)

            # Update the graph
            self.updateGrafik()

    def updateGrafik(self):
        x_vals = np.arange(len(self.data))
        self.curve1.setData(x_vals, self.data)
        self.curve2.setData(x_vals, self.data1)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Real-time Graph Update"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())