
import os
import subprocess
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget, QScrollArea, QPushButton, QAction
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage, QPixmap, QImageReader, QImageWriter
from PyQt5.QtCore import Qt


class RotateImageWindow(QMainWindow):
    def __init__(self, rotated_pixmap):
        super().__init__()
        self.setWindowTitle("Rotated Image")
        self.setWindowState(QtCore.Qt.WindowFullScreen)  # Menjadikan jendela full screen

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)

        layout = QVBoxLayout(central_widget)

        # Create a scroll area
        scroll_area = QScrollArea()
        layout.addWidget(scroll_area)

        # Create a label for the image
        self.rotated_label = QLabel()
        scroll_area.setWidget(self.rotated_label)
        scroll_area.setWidgetResizable(True)  # Enable scrolling if the image is too large

        self.rotated_label.setPixmap(rotated_pixmap)
        self.rotated_label.setAlignment(QtCore.Qt.AlignCenter)

        # Add a "Close" button
        close_button = QPushButton("Close")
        layout.addWidget(close_button)
        close_button.clicked.connect(self.close)

        # Add a "Simpan" button
        save_button = QPushButton("Simpan")
        layout.addWidget(save_button)
        save_button.clicked.connect(self.save_image)


    def closeEvent(self, event):
        self.rotated_label.clear()
        event.accept()

    def save_image(self):
        pixmap = self.rotated_label.pixmap()
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)")
            if file_path:
                pixmap.save(file_path)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(721, 874)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 701, 391))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(10, 410, 701, 391))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(10, 810, 371, 23))
        self.label_3.setFrameShape(QtWidgets.QFrame.Box)
        self.label_3.setText("")
        self.label_3.setObjectName("label_3")
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        self.progressBar.setGeometry(QtCore.QRect(390, 810, 161, 23))
        self.progressBar.setProperty("value", 24)
        self.progressBar.setObjectName("progressBar")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 721, 21))
        self.menubar.setObjectName("menubar")
        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColors)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.menuGeometri = QtWidgets.QMenu(self.menubar)
        self.menuGeometri.setObjectName("menuGeometri")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        MainWindow.setMenuBar(self.menubar)
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionBrightness_Contrast = QtWidgets.QAction(MainWindow)
        self.actionBrightness_Contrast.setObjectName("actionBrightness_Contrast")
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionOpen_File = QtWidgets.QAction(MainWindow)
        self.actionOpen_File.setObjectName("actionOpen_File")
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionRotasi = QtWidgets.QAction(MainWindow)
        self.actionRotasi.setObjectName("actionRotasi")
        self.actionFlip_Horizontal = QtWidgets.QAction(MainWindow)
        self.actionFlip_Horizontal.setObjectName("actionFlip_Horizontal")
        self.actionFlip_Vertikal = QtWidgets.QAction(MainWindow)
        self.actionFlip_Vertikal.setObjectName("actionFlip_Vertikal")
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.actionBrightness_Contrast)
        self.menuColors.addAction(self.actionInvers)
        self.menuGeometri.addAction(self.actionRotasi)
        self.menuGeometri.addAction(self.actionFlip_Horizontal)
        self.menuGeometri.addAction(self.actionFlip_Vertikal)
        self.menuFile.addAction(self.actionOpen_File)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionKeluar)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuGeometri.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuColors.setTitle(_translate("MainWindow", "Colors"))
        self.menuRGB_to_Grayscale.setTitle(_translate("MainWindow", "RGB to Grayscale"))
        self.menuGeometri.setTitle(_translate("MainWindow", "Geometri"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(_translate("MainWindow", "Input Output"))
        self.actionBrightness_Contrast.setText(_translate("MainWindow", "Brightness - Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionOpen_File.setText(_translate("MainWindow", "Open File"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionRotasi.setText(_translate("MainWindow", "Rotasi"))
        self.actionFlip_Horizontal.setText(_translate("MainWindow", "Flip Horizontal"))
        self.actionFlip_Vertikal.setText(_translate("MainWindow", "Flip Vertikal"))
        
        self.progressBar.setValue(0)
        self.actionSave_As.triggered.connect(self.save_image)
        self.actionKeluar.triggered.connect(QtWidgets.qApp.quit)
        self.actionAverage.triggered.connect(self.rgb_to_grayscale_average)
        self.actionLightness.triggered.connect(self.rgb_to_grayscale_lightness)
        self.actionLuminance.triggered.connect(self.rgb_to_grayscale_luminance)
        self.actionInvers.triggered.connect(self.invert_colors)
        self.actionOpen_File.triggered.connect(self.open_image_and_display_path)
        self.actionBrightness_Contrast.triggered.connect(self.open_brightness_contrast_dialog)
        self.actionRotasi.triggered.connect(self.rotate_image)
        self.actionFlip_Horizontal.triggered.connect(self.flip_horizontal)
        self.actionFlip_Vertikal.triggered.connect(self.flip_vertical)

    def rotate_image(self):
        if hasattr(self, 'image'):
            rotation_angle, ok = QtWidgets.QInputDialog.getInt(None, "Rotate Image", "Enter rotation angle (degrees):", 0, -360, 360)
            if ok:
                pixmap = self.label.pixmap()
                if pixmap:
                    # Rotate the pixmap by the specified angle
                    rotated_pixmap = pixmap.transformed(QtGui.QTransform().rotate(rotation_angle))
                    
                    # Show the rotated image in a new window
                    self.rotate_window = RotateImageWindow(rotated_pixmap)
                    self.rotate_window.show()

    def apply_brightness_contrast(self, brightness, contrast):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Create a numpy array to store the adjusted image
            adjusted_image = np.zeros((height, width, 4), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    # Get the RGB pixel values
                    r, g, b, a = QtGui.QColor(image.pixel(x, y)).getRgb()

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

    def open_brightness_contrast_dialog(self):
    # Open a dialog to get user input for brightness and contrast
        brightness, ok1 = QtWidgets.QInputDialog.getInt(None, "Brightness", "Enter brightness (-255 to 255):", 0, -255, 255)
        contrast, ok2 = QtWidgets.QInputDialog.getDouble(None, "Contrast", "Enter contrast (0.01 to 4.0):", 1.0, 0.01, 4.0)

        if ok1 and ok2:
            # Apply brightness and contrast adjustments
            self.apply_brightness_contrast(brightness, contrast)

    def open_image_and_display_path(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.label_2.clear()
        file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
        if file_name:

            image = QtGui.QImage(file_name)
            if not image.isNull():
                pixmap = QtGui.QPixmap.fromImage(image)
                self.label.setPixmap(pixmap)
                self.label.setScaledContents(True)
                self.image = image

    def open_image(self):
            options = QFileDialog.Options()
            options |= QFileDialog.ReadOnly 
            self.label_2.clear()
            file_name, _ = QFileDialog.getOpenFileName(None, "Open Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
            if file_name:
                image = QtGui.QImage(file_name)
                if not image.isNull():
                    pixmap = QtGui.QPixmap.fromImage(image)
                    self.label.setPixmap(pixmap)  
                    self.label.setScaledContents(True) # set  
                    self.image = image 

    def save_image(self):
        pixmap = self.label_2.pixmap()
        if pixmap:
            file_path, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp);;All Files (*)")
            if file_path:
                pixmap.save(file_path)

    def rgb_to_grayscale_average(self, pixmap):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Create a numpy array to store the grayscale image
            grayscale_image = np.zeros((height, width), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    # Get the RGB pixel values
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    
                    # Calculate the average of the RGB values and set it as the grayscale value
                    gray_value = int((r + g + b) / 3)
                    
                    # Set the grayscale pixel value in the numpy array
                    grayscale_image[y][x] = gray_value

            # Create a QImage from the numpy array
            grayscale_qimage = QtGui.QImage(grayscale_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            
            # Create a QPixmap from the QImage and set it to self.label_2
            grayscale_pixmap = QtGui.QPixmap.fromImage(grayscale_qimage)
            self.label_2.setPixmap(grayscale_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def rgb_to_grayscale_lightness(self, pixmap):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Create a numpy array to store the grayscale image
            grayscale_image = np.zeros((height, width), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    # Get the RGB pixel values
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    
                    # Calculate the lightness value
                    lightness_value = int((max(r, g, b) + min(r, g, b)) / 2)
                    
                    # Set the lightness value as the grayscale pixel value in the numpy array
                    grayscale_image[y][x] = lightness_value

            # Create a QImage from the numpy array
            grayscale_qimage = QtGui.QImage(grayscale_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            
            # Create a QPixmap from the QImage and set it to self.label_2
            grayscale_pixmap = QtGui.QPixmap.fromImage(grayscale_qimage)
            self.label_2.setPixmap(grayscale_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def rgb_to_grayscale_luminance(self, pixmap):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Create a numpy array to store the grayscale image
            grayscale_image = np.zeros((height, width), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    # Get the RGB pixel values
                    r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                    
                    # Calculate the luminance (Y) value
                    luminance_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                    
                    # Set the luminance value as the grayscale pixel value in the numpy array
                    grayscale_image[y][x] = luminance_value

            # Create a QImage from the numpy array
            grayscale_qimage = QtGui.QImage(grayscale_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            
            # Create a QPixmap from the QImage and set it to self.label_2
            grayscale_pixmap = QtGui.QPixmap.fromImage(grayscale_qimage)
            self.label_2.setPixmap(grayscale_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def invert_colors(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Create a numpy array to store the inverted image
            inverted_image = np.zeros((height, width, 4), dtype=np.uint8)

            for y in range(height):
                for x in range(width):
                    # Get the RGB pixel values
                    r, g, b, a = QtGui.QColor(image.pixel(x, y)).getRgb()
                    
                    # Invert the color values
                    inverted_r = 255 - r
                    inverted_g = 255 - g
                    inverted_b = 255 - b
                    
                    # Set the inverted color values in the numpy array
                    inverted_image[y][x] = [inverted_r, inverted_g, inverted_b, a]

            # Create a QImage from the numpy array
            inverted_qimage = QtGui.QImage(inverted_image.data, width, height, width * 4, QtGui.QImage.Format_RGBA8888)
            
            # Create a QPixmap from the QImage and set it to self.label_2
            inverted_pixmap = QtGui.QPixmap.fromImage(inverted_qimage)
            self.label_2.setPixmap(inverted_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def flip_horizontal(self):
        width = self.image.width()
        height = self.image.height()

        # Create a numpy array to store the flipped image
        flipped_image = QImage(width, height, QImage.Format_RGBA8888)

        for y in range(height):
            for x in range(width):
                pixel_color = QtGui.QColor(self.image.pixel(x, y))
                flipped_image.setPixelColor(width - 1 - x, y, pixel_color)       

        flipped_pixmap = QPixmap.fromImage(flipped_image)
        self.label_2.setPixmap(flipped_pixmap)
        self.label_2.setAlignment(Qt.AlignCenter)

    def flip_vertical(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            # Create a numpy array to store the flipped image
            flipped_image = QImage(width, height, QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel_color = QtGui.QColor(image.pixel(x, y))
                    flipped_image.setPixelColor(x, height - 1 - y, pixel_color) 

            flipped_pixmap = QPixmap.fromImage(flipped_image)
            self.label_2.setPixmap(flipped_pixmap)
            self.label_2.setAlignment(Qt.AlignCenter)

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())