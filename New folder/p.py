from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QPushButton, QAction
from aritmatika import Ui_Aritmatika as p

import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 366)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 381, 281))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(410, 10, 381, 281))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 300, 241, 16))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        self.label_4.setGeometry(QtCore.QRect(260, 300, 241, 16))
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setText("")
        self.label_4.setObjectName("label_4")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHistogram = QtWidgets.QMenu(self.menuView)
        self.menuHistogram.setObjectName("menuHistogram")
        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuRGB = QtWidgets.QMenu(self.menuColors)
        self.menuRGB.setObjectName("menuRGB")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColors)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.menuBrightness = QtWidgets.QMenu(self.menuColors)
        self.menuBrightness.setObjectName("menuBrightness")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuColors)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuTentang = QtWidgets.QMenu(self.menubar)
        self.menuTentang.setObjectName("menuTentang")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
        self.menuAritmatical_Operation = QtWidgets.QMenu(self.menubar)
        self.menuAritmatical_Operation.setObjectName("menuAritmatical_Operation")
        arithmetic_action = self.menuAritmatical_Operation.addAction("Aritmatika")
        arithmetic_action.triggered.connect(self.frameArimatika)
        self.menuIler = QtWidgets.QMenu(self.menubar)
        self.menuIler.setObjectName("menuIler")
        self.menuEdge_Detection_2 = QtWidgets.QMenu(self.menuIler)
        self.menuEdge_Detection_2.setObjectName("menuEdge_Detection_2")
        self.menuGaussian_Blur = QtWidgets.QMenu(self.menuIler)
        self.menuGaussian_Blur.setObjectName("menuGaussian_Blur")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")
        self.menuMorfologi = QtWidgets.QMenu(self.menubar)
        self.menuMorfologi.setObjectName("menuMorfologi")
        self.menuErosion = QtWidgets.QMenu(self.menuMorfologi)
        self.menuErosion.setObjectName("menuErosion")
        self.menuDilation = QtWidgets.QMenu(self.menuMorfologi)
        self.menuDilation.setObjectName("menuDilation")
        self.menuOpening = QtWidgets.QMenu(self.menuMorfologi)
        self.menuOpening.setObjectName("menuOpening")
        self.menuClosing = QtWidgets.QMenu(self.menuMorfologi)
        self.menuClosing.setObjectName("menuClosing")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionfliphorizontal = QtWidgets.QAction(MainWindow)
        self.actionfliphorizontal.setObjectName("actionFlipHorizontal")
        self.actionfliphorizontal.triggered.connect(self.flip_horizontal)
        self.actionflipvertical = QtWidgets.QAction(MainWindow)
        self.actionflipvertical.setObjectName("actionFlipVertical")
        self.actionflipvertical.triggered.connect(self.flip_vertical)
        self.actionRotate = QtWidgets.QAction(MainWindow)
        self.actionRotate.setObjectName("ActionRotate")
        self.actionRotate.triggered.connect(self.histogram_equalization)
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionOutput.triggered.connect(self.histogram)
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionBrightness_Contrast = QtWidgets.QAction(MainWindow)
        self.actionBrightness_Contrast.setObjectName("actionBrightness_Contrast")
        self.actionBrightness_Contrast.triggered.connect(self.open_brightness_contrast_dialog)
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionGamma_Corretion = QtWidgets.QAction(MainWindow)
        self.actionGamma_Corretion.setObjectName("actionGamma_Corretion")
        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionOrange = QtWidgets.QAction(MainWindow)
        self.actionOrange.setObjectName("actionOrange")
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionPurple = QtWidgets.QAction(MainWindow)
        self.actionPurple.setObjectName("actionPurple")
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionCoklat = QtWidgets.QAction(MainWindow)
        self.actionCoklat.setObjectName("actionCoklat")
        self.actionMerah = QtWidgets.QAction(MainWindow)
        self.actionMerah.setObjectName("actionMerah")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.showAverage)
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(self.showLightness)
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.showLuminance)
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.action1_bit = QtWidgets.QAction(MainWindow)
        self.action1_bit.setObjectName("action1_bit")
        self.action2_bit = QtWidgets.QAction(MainWindow)
        self.action2_bit.setObjectName("action2_bit")
        self.action3_bit = QtWidgets.QAction(MainWindow)
        self.action3_bit.setObjectName("action3_bit")
        self.action4_bit = QtWidgets.QAction(MainWindow)
        self.action4_bit.setObjectName("action4_bit")
        self.action5_bit = QtWidgets.QAction(MainWindow)
        self.action5_bit.setObjectName("action5_bit")
        self.action6_bit = QtWidgets.QAction(MainWindow)
        self.action6_bit.setObjectName("action6_bit")
        self.action7_bit = QtWidgets.QAction(MainWindow)
        self.action7_bit.setObjectName("action7_bit")
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.actionHistogram_Equalization.triggered.connect(self.histogram_equalization)
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_HE_RGB.triggered.connect(self.fuzzy_histogram_equalization_rgb)
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionFuzzy_Grayscale.triggered.connect(self.fuzzy_histogram_equalization_grayscale)
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")
        self.actionAvarage_Filter = QtWidgets.QAction(MainWindow)
        self.actionAvarage_Filter.setObjectName("actionAvarage_Filter")
        self.actionLow_Pass_Filler = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filler.setObjectName("actionLow_Pass_Filler")
        self.actionHight_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHight_Pass_Filter.setObjectName("actionHight_Pass_Filter")
        self.actionBandstop_Filter = QtWidgets.QAction(MainWindow)
        self.actionBandstop_Filter.setObjectName("actionBandstop_Filter")
        self.actionEdge_Detection_1 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_1.setObjectName("actionEdge_Detection_1")
        self.actionEdge_Detection_2 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_2.setObjectName("actionEdge_Detection_2")
        self.actionEdge_Detection_3 = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_3.setObjectName("actionEdge_Detection_3")
        self.actionGaussian_Blur_3_x_3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3_x_3.setObjectName("actionGaussian_Blur_3_x_3")
        self.actionGaussian_Blur_5_X_5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5_X_5.setObjectName("actionGaussian_Blur_5_X_5")
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionSquare_3 = QtWidgets.QAction(MainWindow)
        self.actionSquare_3.setObjectName("actionSquare_3")
        self.actionSquare_5 = QtWidgets.QAction(MainWindow)
        self.actionSquare_5.setObjectName("actionSquare_5")
        self.actionCross_3 = QtWidgets.QAction(MainWindow)
        self.actionCross_3.setObjectName("actionCross_3")
        self.actionSquare_4 = QtWidgets.QAction(MainWindow)
        self.actionSquare_4.setObjectName("actionSquare_4")
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionCross = QtWidgets.QAction(MainWindow)
        self.actionCross.setObjectName("actionCross")
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionOpen_File.triggered.connect(self.openFile)
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionSave_As.triggered.connect(self.saveImage)
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionKeluar.triggered.connect(self.exitApplication)
        self.menuHistogram.addAction(self.actionInput)
        self.menuHistogram.addAction(self.actionOutput)
        self.menuHistogram.addAction(self.actionInput_Output)
        self.menuView.addAction(self.menuHistogram.menuAction())
        self.menuRGB.addAction(self.actionKuning)
        self.menuRGB.addAction(self.actionOrange)
        self.menuRGB.addAction(self.actionCyan)
        self.menuRGB.addAction(self.actionPurple)
        self.menuRGB.addAction(self.actionGray)
        self.menuRGB.addAction(self.actionCoklat)
        self.menuRGB.addAction(self.actionMerah)
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuBrightness.addAction(self.actionContrast)
        self.menuBit_Depth.addAction(self.action1_bit)
        self.menuBit_Depth.addAction(self.action2_bit)
        self.menuBit_Depth.addAction(self.action3_bit)
        self.menuBit_Depth.addAction(self.action4_bit)
        self.menuBit_Depth.addAction(self.action5_bit)
        self.menuBit_Depth.addAction(self.action6_bit)
        self.menuBit_Depth.addAction(self.action7_bit)
        self.menuColors.addAction(self.menuRGB.menuAction())
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.menuBrightness.menuAction())
        self.menuColors.addAction(self.actionBrightness_Contrast)
        self.menuColors.addAction(self.actionInvers)
        self.menuColors.addAction(self.actionLog_Brightness)
        self.menuColors.addAction(self.menuBit_Depth.menuAction())
        self.menuColors.addAction(self.actionGamma_Corretion)
        self.menuImage_Processing.addAction(self.actionHistogram_Equalization)
        self.menuImage_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuImage_Processing.addAction(self.actionFuzzy_Grayscale)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_1)
     
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_2)
        self.menuEdge_Detection_2.addAction(self.actionEdge_Detection_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3_x_3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_5_X_5)
        self.menuIler.addAction(self.actionIdentity)
        self.menuIler.addAction(self.menuEdge_Detection_2.menuAction())
        self.menuIler.addAction(self.actionSharpen)
        self.menuIler.addAction(self.menuGaussian_Blur.menuAction())
        self.menuIler.addAction(self.actionUnsharp_Masking)
        self.menuIler.addAction(self.actionAvarage_Filter)
        self.menuIler.addAction(self.actionLow_Pass_Filler)
        self.menuIler.addAction(self.actionHight_Pass_Filter)
        self.menuIler.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection.addAction(self.actionPrewitt)
        self.menuEdge_Detection.addAction(self.actionSobel)
        self.menuErosion.addAction(self.actionSquare_3)
        self.menuErosion.addAction(self.actionSquare_5)
        self.menuErosion.addAction(self.actionCross_3)
        self.menuDilation.addAction(self.actionSquare_4)
        self.menuDilation.addAction(self.actionSquare_6)
        self.menuDilation.addAction(self.actionCross)
        self.menuOpening.addAction(self.actionSquare_9)
        self.menuClosing.addAction(self.actionSquare_10)
        self.menuMorfologi.addAction(self.menuErosion.menuAction())
        self.menuMorfologi.addAction(self.menuDilation.menuAction())
        self.menuMorfologi.addAction(self.menuOpening.menuAction())
        self.menuMorfologi.addAction(self.menuClosing.menuAction())
        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionKeluar)
        self.menuTentang.addAction(self.actionfliphorizontal)
        self.menuTentang.addAction(self.actionflipvertical)
        self.menuTentang.addAction(self.actionRotate)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuTentang.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuAritmatical_Operation.menuAction())
        self.menubar.addAction(self.menuIler.menuAction())
        self.menubar.addAction(self.menuEdge_Detection.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHistogram.setTitle(_translate("MainWindow", "Histogram"))
        self.menuColors.setTitle(_translate("MainWindow", "Colors"))
        self.menuRGB.setTitle(_translate("MainWindow", "RGB"))
        self.menuRGB_to_Grayscale.setTitle(_translate("MainWindow", "RGB to Grayscale"))
        self.menuBrightness.setTitle(_translate("MainWindow", "Brightness"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuTentang.setTitle(_translate("MainWindow", "Geometri"))
        self.menuImage_Processing.setTitle(_translate("MainWindow", "Image Processing"))
        self.menuAritmatical_Operation.setTitle(_translate("MainWindow", "Aritmatical Operation"))
        self.menuIler.setTitle(_translate("MainWindow", "Filter"))
        self.menuEdge_Detection_2.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuGaussian_Blur.setTitle(_translate("MainWindow", "Gaussian Blur"))
        self.menuEdge_Detection.setTitle(_translate("MainWindow", "Edge Detection"))
        self.menuMorfologi.setTitle(_translate("MainWindow", "Morfologi"))
        self.menuErosion.setTitle(_translate("MainWindow", "Erosion"))
        self.menuDilation.setTitle(_translate("MainWindow", "Dilation"))
        self.menuOpening.setTitle(_translate("MainWindow", "Opening"))
        self.menuClosing.setTitle(_translate("MainWindow", "Closing"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionBrightness_Contrast.setText(_translate("MainWindow", "Brightness - Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionLog_Brightness.setText(_translate("MainWindow", "Log Brightness"))
        self.actionGamma_Corretion.setText(_translate("MainWindow", "Gamma Corretion"))
        self.actionKuning.setText(_translate("MainWindow", "Kuning"))
        self.actionOrange.setText(_translate("MainWindow", "Orange"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionPurple.setText(_translate("MainWindow", "Purple"))
        self.actionGray.setText(_translate("MainWindow", "Gray"))
        self.actionCoklat.setText(_translate("MainWindow", "Coklat"))
        self.actionMerah.setText(_translate("MainWindow", "Merah"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.action1_bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_bit.setText(_translate("MainWindow", "2 bit"))
        self.action3_bit.setText(_translate("MainWindow", "3 bit"))
        self.action4_bit.setText(_translate("MainWindow", "4 bit"))
        self.action5_bit.setText(_translate("MainWindow", "5 bit"))
        self.action6_bit.setText(_translate("MainWindow", "6 bit"))
        self.action7_bit.setText(_translate("MainWindow", "7 bit"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(_translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Grayscale.setText(_translate("MainWindow", "Fuzzy Grayscale"))
        self.actionIdentity.setText(_translate("MainWindow", "Identity"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionUnsharp_Masking.setText(_translate("MainWindow", "Unsharp Masking"))
        self.actionAvarage_Filter.setText(_translate("MainWindow", "Avarage Filter"))
        self.actionLow_Pass_Filler.setText(_translate("MainWindow", "Low Pass Filter"))
        self.actionHight_Pass_Filter.setText(_translate("MainWindow", "Hight Pass Filter"))
        self.actionBandstop_Filter.setText(_translate("MainWindow", "Bandstop Filter"))
        self.actionEdge_Detection_1.setText(_translate("MainWindow", "Edge Detection 1"))
        self.actionEdge_Detection_2.setText(_translate("MainWindow", "Edge Detection 2"))
        self.actionEdge_Detection_3.setText(_translate("MainWindow", "Edge Detection 3"))
        self.actionGaussian_Blur_3_x_3.setText(_translate("MainWindow", "Gaussian Blur 3 x 3"))
        self.actionGaussian_Blur_5_X_5.setText(_translate("MainWindow", "Gaussian Blur 5 X 5 "))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionSquare_3.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_5.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_3.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_4.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 5"))
        self.actionCross.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 9"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionfliphorizontal.setText(_translate("mainWindow" , "Flip Horizontal"))
        self.actionflipvertical.setText(_translate("MainWindow","Flip Vertical"))
        self.actionRotate.setText(_translate("MainWindow","Rotate"))




    def histogram_equalization(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            grayscale_image = np.zeros((height, width), dtype=np.uint8)

            # Menghitung histogram
            histogram = [0] * 256
            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    gray_value = int((r + g + b) / 3)
                    grayscale_image[y][x] = gray_value
                    histogram[gray_value] += 1

            # Menghitung cumulative histogram
            cumulative_histogram = [sum(histogram[:i+1]) for i in range(256)]

            # Normalisasi cumulative histogram
            max_pixel_value = width * height
            normalized_cumulative_histogram = [(cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

            # Menerapkan equalization pada citra
            equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

            equalized_qimage = QtGui.QImage(equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            equalized_pixmap = QtGui.QPixmap.fromImage(equalized_qimage)
            self.label_2.setPixmap(equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

            equalized_qimage = QtGui.QImage(equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            equalized_pixmap = QtGui.QPixmap.fromImage(equalized_qimage)
            self.label_2.setPixmap(equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Buat histogram sebelum equalization
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.hist(np.array(grayscale_image).ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.6)
            plt.title('Histogram Sebelum Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            # Buat histogram sesudah equalization
            plt.subplot(122)
            equalized_image_flat = np.array(equalized_image).ravel()
            plt.hist(equalized_image_flat, bins=256, range=(0, 256), density=True, color='r', alpha=0.6)
            plt.title('Histogram Sesudah Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            plt.tight_layout()
            plt.show()

    def fuzzy_histogram_equalization_grayscale(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            grayscale_image = np.zeros((height, width), dtype=np.uint8)

            # Menghitung histogram
            histogram = [0] * 256
            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    gray_value = int((r + g + b) / 3)
                    grayscale_image[y][x] = gray_value
                    histogram[gray_value] += 1

            # Menghitung cumulative histogram
            cumulative_histogram = [sum(histogram[:i+1]) for i in range(256)]

            # Normalisasi cumulative histogram
            max_pixel_value = width * height
            normalized_cumulative_histogram = [(cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

            # Menerapkan fuzzy equalization pada citra
            fuzzy_equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    fuzzy_equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

            fuzzy_equalized_qimage = QtGui.QImage(fuzzy_equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            fuzzy_equalized_pixmap = QtGui.QPixmap.fromImage(fuzzy_equalized_qimage)
            self.label_2.setPixmap(fuzzy_equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            fuzzy_equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    fuzzy_equalized_image[y][x] = int(normalized_cumulative_histogram[grayscale_image[y][x]])

            fuzzy_equalized_qimage = QtGui.QImage(fuzzy_equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            fuzzy_equalized_pixmap = QtGui.QPixmap.fromImage(fuzzy_equalized_qimage)
            self.label_2.setPixmap(fuzzy_equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Buat histogram sebelum fuzzy equalization
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.hist(np.array(grayscale_image).ravel(), bins=256, range=(0, 256), density=True, color='b', alpha=0.6)
            plt.title('Histogram Sebelum Fuzzy Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            # Buat histogram sesudah fuzzy equalization
            plt.subplot(122)
            fuzzy_equalized_image_flat = np.array(fuzzy_equalized_image).ravel()
            plt.hist(fuzzy_equalized_image_flat, bins=256, range=(0, 256), density=True, color='r', alpha=0.6)
            plt.title('Histogram Sesudah Fuzzy Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            plt.tight_layout()
            plt.show()


    def fuzzy_histogram_equalization_rgb(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Mendefinisikan parameter fuzzy histogram equalization
            m = 3  # Derajat keanggotaan fuzzy
            alpha = 0.2  # Parameter alpha
            beta = 0.7  # Parameter beta

            fuzzy_equalized_image = np.zeros((height, width, 4), dtype=np.uint8)

            # Menghitung histogram per komponen warna
            histograms = [np.zeros(256, dtype=np.uint32) for _ in range(3)]

            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    histograms[0][r] += 1
                    histograms[1][g] += 1
                    histograms[2][b] += 1

            # Menghitung cumulative histogram per komponen warna
            cumulative_histograms = [np.cumsum(hist) for hist in histograms]

            # Normalisasi cumulative histogram
            max_pixel_value = width * height
            normalized_cumulative_histograms = [(cumulative_hist / max_pixel_value) * 255 for cumulative_hist in cumulative_histograms]

            # Menerapkan fuzzy histogram equalization pada citra RGB
            for y in range(height):
                for x in range(width):
                    r, g, b, a = QtGui.QColor(image.pixel(x, y)).getRgb()

                    new_r = int(normalized_cumulative_histograms[0][r])
                    new_g = int(normalized_cumulative_histograms[1][g])
                    new_b = int(normalized_cumulative_histograms[2][b])

                    fuzzy_equalized_image[y][x] = [new_r, new_g, new_b, a]

            fuzzy_equalized_qimage = QtGui.QImage(fuzzy_equalized_image.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888)
            fuzzy_equalized_pixmap = QtGui.QPixmap.fromImage(fuzzy_equalized_qimage)
            self.label_2.setPixmap(fuzzy_equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Buat histogram sebelum fuzzy equalization
            plt.figure(figsize=(12, 6))
            plt.subplot(131)
            hist_channel_r = []
            hist_channel_g = []
            hist_channel_b = []

            for y in range(height):
                for x in range(width):
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    hist_channel_r.append(r)
                    hist_channel_g.append(g)
                    hist_channel_b.append(b)

            plt.hist(hist_channel_r, bins=256, range=(0, 256), density=True, color='r', alpha=0.6, label='R')
            plt.hist(hist_channel_g, bins=256, range=(0, 256), density=True, color='g', alpha=0.6, label='G')
            plt.hist(hist_channel_b, bins=256, range=(0, 256), density=True, color='b', alpha=0.6, label='B')
            plt.title('Histogram Sebelum Fuzzy Equalization (RGB)')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')
            plt.legend()

            # Buat histogram sesudah fuzzy equalization
            plt.subplot(132)
            fuzzy_equalized_image_flat = np.array(fuzzy_equalized_image).reshape(-1, 4)
            hist_channel_r = fuzzy_equalized_image_flat[:, 0]
            hist_channel_g = fuzzy_equalized_image_flat[:, 1]
            hist_channel_b = fuzzy_equalized_image_flat[:, 2]

            plt.hist(hist_channel_r, bins=256, range=(0, 256), density=True, color='r', alpha=0.6, label='R')
            plt.hist(hist_channel_g, bins=256, range=(0, 256), density=True, color='g', alpha=0.6, label='G')
            plt.hist(hist_channel_b, bins=256, range=(0, 256), density=True, color='b', alpha=0.6, label='B')
            plt.title('Histogram Sesudah Fuzzy Equalization (RGB)')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')
            plt.legend()

            plt.tight_layout()
            plt.show()

    def frameArimatika(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = p()
        self.ui.setupUi(self.window)
        self.window.show() 
    
        
    def saveImage(self):
        # Inisialisasi opsi untuk dialog pemilihan berkas
        options = QFileDialog.Options()
        # Menambahkan opsi mode baca saja ke dalam opsi dialog
        options |= QFileDialog.ReadOnly 
        # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        # check apakah terdapat path file
        if file_name:
            #Simpan gambar yang telah diformat
            pixmap = self.label_2.pixmap()
            pixmap.save(file_name)
            self.label_4.setText(file_name)
            
    def rotateImage(self):
        rotation , ok = QtWidgets.QInputDialog.getInt(None , "Rotate Image","Enter rotation angle (degress):",0,-360,360)
        if ok:
            current_pixmap = self.label.pixmap()
            second_pixmap = self.label_2.pixmap()
            
            if self.label_2.pixmap() is None:
                    rotated_image = current_pixmap.transformed(QtGui.QTransform().rotate(rotation))
                    # rotated_pixmap = QtGui.QPixmap.fromImage(rotated_image)
                    self.label_2.setPixmap(rotated_image)
                    self.label_2.setAlignment(QtCore.Qt.AlignCenter)
                    self.label_2.setScaledContents(True)
                    self.image = rotated_image.toImage()
                    
            else:    
                    rotated_image = second_pixmap.transformed(QtGui.QTransform().rotate(rotation))
                    # rotated_pixmap = QtGui.QPixmap.fromImage(rotated_image)
                    self.label_2.setPixmap(rotated_image)
                    self.label_2.setAlignment(QtCore.Qt.AlignCenter)
                    self.label_2.setScaledContents(True)
                    self.image = rotated_image.toImage()
                                   
                
    def showAverage(self):
        width = self.image.width()
        height = self.image.height()
        average = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusAverage = int((r + g + b)/3)
                grayscale_color = QtGui.QColor(rumusAverage, rumusAverage, rumusAverage)
                average.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(average) 
        self.label_2.setPixmap(p)
        self.image = average
    
    def edge_detection(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Mengkonversi gambar QtImage ke format PIL
            image_pil = Image.fromqpixmap(image)

            # Konversi gambar ke grayscale
            grayscale_image = image_pil.convert('L')

            # Deteksi tepi dengan filter Sobel
            edge_image = grayscale_image.filter(ImageFilter.FIND_EDGES)

            # Mengonversi kembali ke format QtImage
            edge_image_qt = QtGui.QPixmap.fromImage(QtGui.QImage(edge_image.tobytes(), width, height, QtGui.QImage.Format_Grayscale8))
            self.label_2.setPixmap(edge_image_qt)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Menampilkan citra sebelum edge detection
            self.label.setPixmap(pixmap)
            self.label.setAlignment(QtCore.Qt.AlignCenter)


    def showLuminance(self):
        width = self.image.width()
        height = self.image.height()
        grayscale_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLuminance = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_color = QtGui.QColor(rumusLuminance, rumusLuminance, rumusLuminance)
                grayscale_image.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(grayscale_image) 
        self.label_2.setPixmap(p)
        self.image = grayscale_image
        
    def openFile(self):
        options = QFileDialog.Options() # Inisialisasi opsi untuk dialog pemilihan berkas
        options |= QFileDialog.ReadOnly # Menambahkan opsi mode baca saja ke dalam opsi dialog
        # menghapus gambar dari label kedua
        self.label_2.clear()
        self.label_3.setText("")
        self.label_4.setText("")
        # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        # check apakah terdapat path file
        if file_name:
            # untuk membuat objek QImage dari suatu berkas gambar dengan nama file_name
            image = QtGui.QImage(file_name)
            # check varibale apakah tidak kosong
            if not image.isNull():
                # simpan gambar pada variable pixmap
                pixmap = QtGui.QPixmap.fromImage(image)
                self.label.setPixmap(pixmap)  # Set gambar pada label
                self.label.setScaledContents(True) # set  kontennya agar sesuai dengan ukuran label.
                self.image = image # menetapkan objek QImage yang sudah dibuat sebelumnya (dalam contoh kode sebelumnya) ke atribut self.image dari kelas atau objek saat ini
                self.checkHisto = file_name
                self.label_3.setText(file_name)
                       
    def bit_depth(self, new_bit_depth):
        if new_bit_depth < 1 or new_bit_depth > 8:
            print("Kedalaman bit yang diminta harus antara 1 hingga 8.")
            return

        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Menghitung maksimal nilai piksel sesuai dengan kedalaman bit yang diinginkan
            max_pixel_value = 2 ** new_bit_depth - 1

            # Mengonversi citra menjadi array NumPy
            image_array = QtGui.QImage(image).convertToFormat(QtGui.QImage.Format_Grayscale8)
            image_np = np.array(image_array)

            # Mengubah kedalaman bit citra
            bit_depth_scale = 255 / max_pixel_value
            image_scaled = (image_np / bit_depth_scale).astype(np.uint8)
            image_scaled = (image_scaled * bit_depth_scale).astype(np.uint8)

            # Mengonversi kembali ke format QImage
            new_qimage = QtGui.QImage(image_scaled.data, width, height, QtGui.QImage.Format_Grayscale8)
            new_pixmap = QtGui.QPixmap.fromImage(new_qimage)
            self.label_2.setPixmap(new_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)  
    def showLightness(self):
        width = self.image.width()
        height = self.image.height()
        lightness = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)
        
        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                rumusLightness = int((max(r,g,b)+ min(r,g,b))/2)
                grayscale_color = QtGui.QColor(rumusLightness, rumusLightness, rumusLightness)
                lightness.setPixelColor(x, y, grayscale_color)
        
        p = QtGui.QPixmap.fromImage(lightness) 
        self.label_2.setPixmap(p)
        self.image = lightness
        
    def histogram(self):
        img = cv2.imread(self.checkHisto , 1)
        # alternative way to find histogram of an image
        plt.hist(img.ravel(),256,[0,256])
        plt.show()
    def invert(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            inverted_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel = QtGui.QColor(image.pixel(x, y))
                    inverted_color = QtGui.QColor(255 - pixel.red(), 255 - pixel.green(), 255 - pixel.blue(), pixel.alpha())
                    inverted_image.setPixel(x, y, inverted_color.rgba())

            inverted_pixmap = QtGui.QPixmap.fromImage(inverted_image)
            self.label_2.setPixmap(inverted_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
    def apply_brightness_contrast(self, brightness, contrast):
        width = self.image.width()
        height = self.image.height()
        # Create a numpy array to store the adjusted image
        adjusted_image = np.zeros((height, width, 4), dtype=np.uint8)

        for y in range(height):
            for x in range(width):
                # Get the RGB pixel values
                r, g, b, a = QtGui.QColor(self.image.pixel(x, y)).getRgb()
                # Adjust brightness
                adjusted_r = min(max(r + brightness, 0), 255)
                adjusted_g = min(max(g + brightness, 0), 255)
                adjusted_b = min(max(b + brightness, 0), 255)
                # Adjust contrast
                adjusted_r = min(max(((adjusted_r - 127) * contrast) + 127, 0), 255)
                adjusted_g = min(max(((adjusted_g - 127) * contrast) + 127, 0), 255)
                adjusted_b = min(max(((adjusted_b - 127) * contrast) + 127, 0), 255)

                # Set the adjusted color values in the numpy array
                adjusted_image[y][x] = [adjusted_r, adjusted_g, adjusted_b, a]

        # Create a QImage from the numpy array
        adjusted_qimage = QtGui.QImage(adjusted_image.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888)

        # Create a QPixmap from the QImage and set it to self.label_2
        adjusted_pixmap = QtGui.QPixmap.fromImage(adjusted_qimage)
        self.label_2.setPixmap(adjusted_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.image = adjusted_qimage

    def open_brightness_contrast_dialog(self):
    # Open a dialog to get user input for brightness and contrast
        brightness, ok1 = QtWidgets.QInputDialog.getInt(None, "Brightness", "Enter brightness (-255 to 255):", 0, -255, 255)
        contrast, ok2 = QtWidgets.QInputDialog.getDouble(None, "Contrast", "Enter contrast (0.01 to 4.0):", 1.0, 0.01, 4.0)

        if ok1 and ok2:
            # Apply brightness and contrast adjustments
            self.apply_brightness_contrast(brightness, contrast)
            
    def flip_horizontal(self):
            width = self.image.width()
            height = self.image.height()

            # Create a numpy array to store the flipped image
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))
                    flipped_image.setPixelColor(width - 1 - x, y, pixel_color)       

            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.image = flipped_image
            
    def flip_vertical(self):
            width = self.image.width()
            height = self.image.height()

            # Create a numpy array to store the flipped image
            flipped_image = QtGui.QImage(width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(self.image.pixel(x, y))
                    flipped_image.setPixelColor(x, height - 1 - y, pixel_color) 

            flipped_pixmap = QtGui.QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            self.image = flipped_image        
             
    def exitApplication(self):
        # untuk keluar dari aplikasi
        QtWidgets.qApp.quit()    
        
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
