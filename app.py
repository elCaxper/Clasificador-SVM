#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Qt requirements"""

import os

from PySide.QtGui import QMainWindow, QApplication, QLabel, QPixmap
from PySide.QtGui import QAction, QMenu, QMessageBox, QComboBox
from PySide import QtCore, QtGui  # NEW WINDOW

from ventana_principal import Ui_MainWindow

import sys



class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)




if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    app.exec_()