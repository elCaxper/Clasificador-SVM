# -*- coding: utf-8 -*-

from PySide import QtCore, QtGui
import pandas as pd
from os import listdir
from os.path import isfile, join
import os.path
import numpy as np

from sklearn import svm, metrics
from sklearn.externals import joblib

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(564, 192)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setMaximumSize(QtCore.QSize(640, 435))
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtGui.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtGui.QHBoxLayout()
        self.horizontalLayout.setContentsMargins(0, -1, 0, -1)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout_2 = QtGui.QVBoxLayout()
        self.verticalLayout_2.setSpacing(0)
        self.verticalLayout_2.setSizeConstraint(QtGui.QLayout.SetDefaultConstraint)
        self.verticalLayout_2.setContentsMargins(0, 0, -1, -1)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.btn_entrenar = QtGui.QPushButton(self.centralwidget)
        self.btn_entrenar.setObjectName("btn_entrenar")
        self.verticalLayout_2.addWidget(self.btn_entrenar)
        self.btn_clasificar = QtGui.QPushButton(self.centralwidget)
        self.btn_clasificar.setObjectName("btn_clasificar")
        self.verticalLayout_2.addWidget(self.btn_clasificar)

        self.btn_clasificar_folder = QtGui.QPushButton(self.centralwidget)
        self.btn_clasificar_folder.setObjectName("btn_clasificar_folder")
        self.verticalLayout_2.addWidget(self.btn_clasificar_folder)

        self.btn_detalles = QtGui.QPushButton(self.centralwidget)
        self.btn_detalles.setObjectName("btn_detalles")
        self.verticalLayout_2.addWidget(self.btn_detalles)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        spacerItem = QtGui.QSpacerItem(30, 20, QtGui.QSizePolicy.Fixed, QtGui.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem)
        self.verticalLayout_3 = QtGui.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.gridLayout_3 = QtGui.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.cb_kernel = QtGui.QComboBox(self.centralwidget)
        self.cb_kernel.setObjectName("cb_kernel")
        self.gridLayout_3.addWidget(self.cb_kernel, 0, 1, 1, 1)
        self.lb_kernel = QtGui.QLabel(self.centralwidget)
        self.lb_kernel.setObjectName("lb_kernel")
        self.gridLayout_3.addWidget(self.lb_kernel, 0, 0, 1, 1)
        self.lb_degree = QtGui.QLabel(self.centralwidget)
        self.lb_degree.setObjectName("lb_degree")
        self.gridLayout_3.addWidget(self.lb_degree, 1, 0, 1, 1)
        self.cb_degree = QtGui.QComboBox(self.centralwidget)
        self.cb_degree.setObjectName("cb_degree")
        self.gridLayout_3.addWidget(self.cb_degree, 1, 1, 1, 1)
        self.lb_gamma = QtGui.QLabel(self.centralwidget)
        self.lb_gamma.setObjectName("lb_gamma")
        self.gridLayout_3.addWidget(self.lb_gamma, 3, 0, 1, 1)
        self.lineEdit = QtGui.QLineEdit(self.centralwidget)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_3.addWidget(self.lineEdit, 3, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_3)
        self.horizontalLayout.addLayout(self.verticalLayout_3)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 564, 23))
        self.menubar.setObjectName("menubar")
        self.menuArchivo = QtGui.QMenu(self.menubar)
        self.menuArchivo.setObjectName("menuArchivo")
        self.menuSVM = QtGui.QMenu(self.menuArchivo)
        self.menuSVM.setObjectName("menuSVM")
        self.menuAyuda = QtGui.QMenu(self.menubar)
        self.menuAyuda.setObjectName("menuAyuda")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionAbrir = QtGui.QAction(MainWindow)
        self.actionAbrir.setObjectName("actionAbrir")
        self.actionAcerca_de = QtGui.QAction(MainWindow, triggered=self.about)
        self.actionAcerca_de.setObjectName("actionAcerca_de")
        self.actionCargar_SVM = QtGui.QAction(MainWindow)
        self.actionCargar_SVM.setObjectName("actionCargar_SVM")
        self.actionGuardar_SVM = QtGui.QAction(MainWindow)
        self.actionGuardar_SVM.setObjectName("actionGuardar_SVM")
        self.menuSVM.addAction(self.actionCargar_SVM)
        self.menuSVM.addAction(self.actionGuardar_SVM)
        self.menuArchivo.addAction(self.actionAbrir)
        self.menuArchivo.addAction(self.menuSVM.menuAction())
        self.menuAyuda.addAction(self.actionAcerca_de)
        self.menubar.addAction(self.menuArchivo.menuAction())
        self.menubar.addAction(self.menuAyuda.menuAction())

        self.btn_clasificar.setDisabled(True)
        self.btn_clasificar_folder.setDisabled(True)

        self.btn_detalles.setDisabled(True)
        self.btn_detalles.clicked.connect(self.resultados)
        self.btn_entrenar.setDisabled(True)
        self.btn_entrenar.clicked.connect(self.entrenar)
        self.btn_clasificar.clicked.connect(self.onInputFileButtonClicked)
        self.btn_clasificar_folder.clicked.connect(self.selectDirectory_folder)
        self.actionAbrir.triggered.connect(self.selectDirectory)

        self.actionGuardar_SVM.triggered.connect(self.guardar_svm)
        self.actionCargar_SVM.triggered.connect(self.cargar_svm)

        self.lista_kernels = [
            'Linear',
            'Polynomial',
            'RBF',
            'Sigmoid'
        ]

        self._lista_kernels = [
            'linear',
            'poly',
            'rbf',
            'sigmoid'
        ]

        self.cb_kernel.addItems(self.lista_kernels)
        self.lineEdit.setText('0.001')

        self.cb_kernel.currentIndexChanged.connect(self.cb_kernel_change)
        self.cb_degree.currentIndexChanged.connect(self.cb_degree_change)
        self.cb_degree.setDisabled(True)
        self.lineEdit.setDisabled(True)
        self.lb_gamma.setText('Gamma')
        self.kernel_slected = 0

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def selectDirectory(self):
        self.selected_directory = QtGui.QFileDialog.getExistingDirectory()

        if self.selected_directory == '':
            print('directorio incorrecto incorrecto')

        else:

            num_dat_files = [f.split('.')[1] for f in sorted(listdir(self.selected_directory)) if
                             isfile(join(self.selected_directory, f)) and f.endswith(".dat")].count('dat')

            if num_dat_files > 0:

                onlyfiles = [f.split('1')[0] for f in sorted(listdir(self.selected_directory)) if
                             isfile(join(self.selected_directory, f)) and f.endswith(".dat")]
                self.target = np.array(onlyfiles)

                self.num_labels = len(set(onlyfiles))

                print('selected_directory:', self.selected_directory)
                print('num_dat_files:', num_dat_files)
                print('num_labels:', self.num_labels)

                self.cb_degree.clear()
                if self.num_labels + 4 < 10:
                    for i in range(self.num_labels, self.num_labels + 4):
                        self.cb_degree.addItem(str(i))
                else:
                    for i in range(5, 10):
                        self.cb_degree.addItem(str(i))

                l = [pd.read_table(join(self.selected_directory, filename), delim_whitespace=True,
                                   names=('X', 'Y', 'Value'))
                     ['Value']
                     for filename in sorted(listdir(self.selected_directory)) if
                     (isfile(join(self.selected_directory, filename)) and filename.endswith('.dat'))]
                # df = pd.concat(l, axis=0)
                self.data = np.array(l)
                print('Tamanio', self.data.size)

                self.btn_entrenar.setDisabled(False)

    def selectDirectory_folder(self):
        self.selected_directory_test = QtGui.QFileDialog.getExistingDirectory()

        if self.selected_directory_test == '':
            print('directorio incorrecto incorrecto')

        else:

            num_dat_files = [f.split('.')[1] for f in sorted(listdir(self.selected_directory_test)) if
                             isfile(join(self.selected_directory_test, f)) and f.endswith(".dat")].count('dat')

            if num_dat_files > 0:

                onlyfiles = [f.split('1')[0] for f in sorted(listdir(self.selected_directory_test)) if
                             isfile(join(self.selected_directory_test, f)) and f.endswith(".dat")]
                target = np.array(onlyfiles)

                num_labels = len(set(onlyfiles))

                # Use the selected directory...
                print('selected_directory:', self.selected_directory_test)
                print('num_dat_files:', num_dat_files)
                print('num_labels:', num_labels)


                l = [pd.read_table(join(self.selected_directory_test, filename), delim_whitespace=True,
                                   names=('X', 'Y', 'Value'))
                     ['Value']
                     for filename in sorted(listdir(self.selected_directory_test)) if
                     (isfile(join(self.selected_directory_test, filename)) and filename.endswith('.dat'))]
                # df = pd.concat(l, axis=0)
                data = np.array(l)
                print('Tamanio', data.size)

                self.expected = target[:]
                self.predicted = self.classifier.predict(data[:])
                print("Classification report for classifier %s:\n%s\n"
                      % (self.classifier, metrics.classification_report(self.expected, self.predicted)))

                print('expected', self.expected)
                print('predicted', self.predicted)

                self.btn_detalles.setDisabled(False)


    def about(self):
        QtGui.QMessageBox.about(self, "Acerda de",
                                "Programa Creado por Gustavo Plaza Roma "
                                "para la asignatura minería de datos.")

    def resultados(self):
        QtGui.QMessageBox.about(self, "Resultados",
                                "{} \n Esperado: {}, Predicho: {} ".format(metrics.classification_report(self.expected, self.predicted),
                                                   self.expected[0],self.predicted[0]),
                                )


    def cb_kernel_change(self, string):
        self.kernel_slected = string
        if string == 0:
            self.cb_degree.setDisabled(True)
            self.lineEdit.setDisabled(True)
            self.lb_gamma.setText('Gamma')
        elif string == 1:
            self.cb_degree.setDisabled(False)
            self.lb_gamma.setText('coef0')
            self.lineEdit.setDisabled(False)
            self.lineEdit.setText(str(0.0))
        elif string == 2:
            self.cb_degree.setDisabled(True)
            self.lineEdit.setDisabled(False)
            self.lb_gamma.setText('Gamma')
            self.lineEdit.setText(str(0.0001))
        elif string == 3:
            self.cb_degree.setDisabled(True)
            self.lineEdit.setDisabled(False)
            self.lb_gamma.setText('coef0')
            self.lineEdit.setText(str(0.0))

    def entrenar(self):

        _kernel = self._lista_kernels[self.cb_kernel.currentIndex()]
        _degree = self.degree_slected + self.num_labels

        print('kernel', _kernel)
        print('degree', _degree)
        if self.cb_kernel.currentIndex() == 1:
            _coef0 = float(self.lineEdit.text())
            self.classifier = svm.SVC(coef0=_coef0, degree=_degree,
                                      kernel=_kernel)
        else:
            _gamma = float(self.lineEdit.text())
            self.classifier = svm.SVC(gamma=_gamma, kernel=_kernel)

        self.classifier.fit(self.data[:round(len(self.data))], self.target[:round(len(self.data))])

        self.btn_clasificar.setDisabled(False)
        self.btn_clasificar_folder.setDisabled(False)

        print("Entrenado")


    def cb_degree_change(self, string):
        self.degree_slected = int(string)

    def onInputFileButtonClicked(self):
        filename, filter = QtGui.QFileDialog.getOpenFileName(parent=self, caption='Open file', dir='.',
                                                             filter='*.dat')
        head, tail = os.path.split(filename)
        target = np.array([tail.split('1')[0]])

        print('filename', target)

        if filename!= '':

            l = [pd.read_table(filename, delim_whitespace=True, names=('X', 'Y', 'Value'))
                 ['Value']]
            # df = pd.concat(l, axis=0)
            data = np.array(l)
            print('Tamanio', data.size)

            test = data[0].reshape(1, -1)
            print(test)
            self.expected = target
            self.predicted = self.classifier.predict(test)
            print("Classification report for classifier %s:\n%s\n"
                  % (self.classifier, metrics.classification_report(self.expected, self.predicted)))

            print('expected', self.expected)
            print('predicted', self.predicted)

            self.btn_detalles.setDisabled(False)

    def cargar_svm(self):
        filename, filter = QtGui.QFileDialog.getOpenFileName(parent=self, caption='Open file', dir='.',
                                                             filter='*.pkl')

        if filename!='':
            self.classifier = joblib.load(filename)
            self.btn_clasificar.setDisabled(False)
            self.btn_clasificar_folder.setDisabled(False)

    def guardar_svm(self):
        fileName = QtGui.QFileDialog.getSaveFileName(self, 'Dialog Title', './',
                                                     selectedFilter='*.pkl')
        if fileName:
            print(fileName[0])

            joblib.dump(self.classifier, fileName[0])

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(
            QtGui.QApplication.translate("MainWindow", "Clasificador SVM", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_entrenar.setText(
            QtGui.QApplication.translate("MainWindow", "Entrenar", None, QtGui.QApplication.UnicodeUTF8))
        self.btn_clasificar.setText(
            QtGui.QApplication.translate("MainWindow", "Clasificar", None, QtGui.QApplication.UnicodeUTF8))

        self.btn_clasificar_folder.setText(
            QtGui.QApplication.translate("MainWindow", "Clasificar Carpeta", None, QtGui.QApplication.UnicodeUTF8))

        self.btn_detalles.setText(
            QtGui.QApplication.translate("MainWindow", "Ver detalles", None, QtGui.QApplication.UnicodeUTF8))
        self.lb_kernel.setText(
            QtGui.QApplication.translate("MainWindow", "Kernel:", None, QtGui.QApplication.UnicodeUTF8))
        self.lb_degree.setText(
            QtGui.QApplication.translate("MainWindow", "Grado:", None, QtGui.QApplication.UnicodeUTF8))
        self.lb_gamma.setText(
            QtGui.QApplication.translate("MainWindow", "Gamma:", None, QtGui.QApplication.UnicodeUTF8))
        self.menuArchivo.setTitle(
            QtGui.QApplication.translate("MainWindow", "Menu", None, QtGui.QApplication.UnicodeUTF8))
        self.menuSVM.setTitle(QtGui.QApplication.translate("MainWindow", "SVM", None, QtGui.QApplication.UnicodeUTF8))
        self.menuAyuda.setTitle(
            QtGui.QApplication.translate("MainWindow", "Ayuda", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAbrir.setText(
            QtGui.QApplication.translate("MainWindow", "Abrir Train Dir", None, QtGui.QApplication.UnicodeUTF8))
        self.actionAcerca_de.setText(
            QtGui.QApplication.translate("MainWindow", "Acerca de", None, QtGui.QApplication.UnicodeUTF8))
        self.actionCargar_SVM.setText(
            QtGui.QApplication.translate("MainWindow", "Cargar SVM", None, QtGui.QApplication.UnicodeUTF8))
        self.actionGuardar_SVM.setText(
            QtGui.QApplication.translate("MainWindow", "Guardar SVM", None, QtGui.QApplication.UnicodeUTF8))
