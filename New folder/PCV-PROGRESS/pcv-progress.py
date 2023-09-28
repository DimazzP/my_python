from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap, QImage, QColor
from aritmatics import Ui_Aritmatics as p

import matplotlib.pyplot as plt
import numpy as np

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):

    def openImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_name:
            try:
                image = QtGui.QPixmap(file_name)
                label_width = self.label.width()
                label_height = self.label.height()
                scaled_image = image.scaled(
                    label_width, label_height, QtCore.Qt.KeepAspectRatio)
                self.label.setPixmap(scaled_image)
                self.label.setAlignment(QtCore.Qt.AlignCenter)
                self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    None, "Error", f"Error opening image: {str(e)}")

    def saveAsImage(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(None, "Save Image As",
                                                   "", "Images (*.png *.jpg *.jpeg *.bmp)", options=options)
        if file_name:
            pixmap = self.label_2.pixmap()
            pixmap.save(file_name)

    def exitApplication(self):
        QtWidgets.QApplication.quit()

    def averageGrayscale(self):
        pixmap = self.label.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        grayscale_image = QImage(width, height, QImage.Format_RGB888)

        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                gray_value = int((r + g + b) / 3)
                grayscale_color = QColor(gray_value, gray_value, gray_value)
                grayscale_image.setPixelColor(x, y, grayscale_color)

        pixmap_grayscale = QPixmap.fromImage(grayscale_image)
        self.label_2.setPixmap(pixmap_grayscale)

    def lightnessGrayscale(self):
        pixmap = self.label.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        grayscale_image = QImage(width, height, QImage.Format_RGB888)

        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                gray_value = int((max(r, g, b) + min(r, g, b)) / 2)
                grayscale_color = QColor(gray_value, gray_value, gray_value)
                grayscale_image.setPixelColor(x, y, grayscale_color)

        pixmap_grayscale = QPixmap.fromImage(grayscale_image)
        self.label_2.setPixmap(pixmap_grayscale)

    def luminanceGrayscale(self):
        pixmap = self.label.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        grayscale_image = QImage(width, height, QImage.Format_RGB888)

        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                grayscale_luminance = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_color = QColor(
                    grayscale_luminance, grayscale_luminance, grayscale_luminance)
                grayscale_image.setPixelColor(x, y, grayscale_color)

        pixmap_grayscale = QPixmap.fromImage(grayscale_image)
        self.label_2.setPixmap(pixmap_grayscale)

    def flipHorizontal(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            flipped_pixmap = original_pixmap.transformed(
                QtGui.QTransform().scale(-1, 1))
            self.label_2.setPixmap(flipped_pixmap)

    def flipVertical(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            flipped_pixmap = original_pixmap.transformed(
                QtGui.QTransform().scale(1, -1))
            self.label_2.setPixmap(flipped_pixmap)

    def rotasi(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            rotated_pixmap = original_pixmap.transformed(
                QtGui.QTransform().rotate(90))
            self.label_2.setPixmap(rotated_pixmap)

    def translasi(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            delta_x, _ = QtWidgets.QInputDialog.getInt(
                None, "Translate Image", "Enter Delta X:")
            delta_y, _ = QtWidgets.QInputDialog.getInt(
                None, "Translate Image", "Enter Delta Y:")

        translated_pixmap = QtGui.QPixmap(original_pixmap)
        painter = QtGui.QPainter(translated_pixmap)
        painter.translate(delta_x, delta_y)
        painter.drawPixmap(0, 0, original_pixmap)
        painter.end()

        self.label_2.setPixmap(translated_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

    def uniformScaling(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            scale_factor, _ = QtWidgets.QInputDialog.getDouble(
                None, "Uniform Scaling", "Enter scaling factor:")
            if scale_factor > 0:
                scaled_pixmap = original_pixmap.scaled(
                    original_pixmap.size() * scale_factor, QtCore.Qt.KeepAspectRatio)
                self.label_2.setPixmap(scaled_pixmap)
                self.label_2.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Invalid Input", "Please enter a positive scaling factor.")

    def nonUniformScaling(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            scale_factor_x, _ = QtWidgets.QInputDialog.getDouble(
                None, "Non-Uniform Scaling", "Enter X scaling factor:")
            scale_factor_y, _ = QtWidgets.QInputDialog.getDouble(
                None, "Non-Uniform Scaling", "Enter Y scaling factor:")

        if scale_factor_x > 0 and scale_factor_y > 0:
            width = int(original_pixmap.width() * scale_factor_x)
            height = int(original_pixmap.height() * scale_factor_y)
            scaled_pixmap = original_pixmap.scaled(
                width, height, QtCore.Qt.KeepAspectRatio)
            self.label_2.setPixmap(scaled_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Invalid Input", "Please enter positive scaling factors for both X and Y.")

    def histogramEqualization(self):
        original_pixmap = self.label.pixmap()
        if original_pixmap:
            # Konversi pixmap ke QImage
            image = original_pixmap.toImage()
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
            normalized_cumulative_histogram = [
                (cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

            # Menerapkan equalization pada citra
            equalized_image = np.zeros((height, width), dtype=np.uint8)
            for y in range(height):
                for x in range(width):
                    equalized_image[y][x] = int(
                        normalized_cumulative_histogram[grayscale_image[y][x]])

            equalized_qimage = QtGui.QImage(
                equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
            equalized_pixmap = QtGui.QPixmap.fromImage(equalized_qimage)
            self.label_2.setPixmap(equalized_pixmap)
            self.label_2.setAlignment(QtCore.Qt.AlignCenter)

            # Buat histogram sebelum equalization
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.hist(np.array(grayscale_image).ravel(), bins=256,
                     range=(0, 256), density=True, color='b', alpha=0.6)
            plt.title('Histogram Sebelum Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            # Buat histogram sesudah equalization
            plt.subplot(122)
            equalized_image_flat = np.array(equalized_image).ravel()
            plt.hist(equalized_image_flat, bins=256, range=(
                0, 256), density=True, color='r', alpha=0.6)
            plt.title('Histogram Sesudah Equalization')
            plt.xlabel('Nilai Pixel')
            plt.ylabel('Frekuensi Relatif')

            plt.tight_layout()
            plt.show()

    def fuzzyHERGB(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

        equalized_image = QtGui.QImage(
            width, height, QtGui.QImage.Format_RGB32)

        # Menghitung histogram untuk setiap saluran warna
        histograms = [np.zeros(256, dtype=int) for _ in range(3)]
        cumulative_histograms = [np.zeros(256, dtype=int) for _ in range(3)]

        for y in range(height):
            for x in range(width):
                r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                histograms[0][r] += 1
                histograms[1][g] += 1
                histograms[2][b] += 1

        # Menghitung cumulative histogram untuk setiap saluran warna
        for i in range(3):
            cumulative_histograms[i][0] = histograms[i][0]
            for j in range(1, 256):
                cumulative_histograms[i][j] = cumulative_histograms[i][j -
                                                                       1] + histograms[i][j]

        # Normalisasi cumulative histogram
        max_pixel_value = width * height
        normalized_cumulative_histograms = [
            cumulative_histograms[i] / max_pixel_value * 255 for i in range(3)]

        # Menerapkan fuzzy equalization pada citra
        for y in range(height):
            for x in range(width):
                r, g, b, _ = QtGui.QColor(image.pixel(x, y)).getRgb()
                new_r = int(normalized_cumulative_histograms[0][r])
                new_g = int(normalized_cumulative_histograms[1][g])
                new_b = int(normalized_cumulative_histograms[2][b])
                equalized_image.setPixel(x, y, QtGui.qRgb(new_r, new_g, new_b))

        equalized_pixmap = QtGui.QPixmap.fromImage(equalized_image)
        self.label_2.setPixmap(equalized_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

        # Buat histogram sebelum fuzzy equalization
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        for i, color in enumerate(['r', 'g', 'b']):
            plt.hist(np.array([QtGui.QColor(image.pixel(x, y)).getRgb()[i] for x in range(width) for y in range(height)]),
                     bins=256, range=(0, 256), density=True, color=color, alpha=0.6, label=color.upper())
        plt.title('Histogram Sebelum Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')
        plt.legend()

        # Buat histogram sesudah fuzzy equalization
        plt.subplot(122)
        for i, color in enumerate(['r', 'g', 'b']):
            plt.hist(np.array([QtGui.QColor(equalized_image.pixel(x, y)).getRgb()[i] for x in range(width) for y in range(height)]),
                     bins=256, range=(0, 256), density=True, color=color, alpha=0.6, label=color.upper())
        plt.title('Histogram Sesudah Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def fuzzyGreyscale(self):
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
                pixel_value = QtGui.qGray(image.pixel(x, y))
                grayscale_image[y][x] = pixel_value
                histogram[pixel_value] += 1

        # Menghitung cumulative histogram
        cumulative_histogram = [sum(histogram[:i+1]) for i in range(256)]

        # Normalisasi cumulative histogram
        max_pixel_value = width * height
        normalized_cumulative_histogram = [
            (cumulative_histogram[i] / max_pixel_value) * 255 for i in range(256)]

        # Menerapkan fuzzy equalization pada citra grayscale
        fuzzy_equalized_image = np.zeros((height, width), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                fuzzy_equalized_image[y][x] = int(
                    normalized_cumulative_histogram[grayscale_image[y][x]])

        fuzzy_equalized_qimage = QtGui.QImage(
            fuzzy_equalized_image.data, width, height, width, QtGui.QImage.Format_Grayscale8)
        fuzzy_equalized_pixmap = QtGui.QPixmap.fromImage(
            fuzzy_equalized_qimage)
        self.label_2.setPixmap(fuzzy_equalized_pixmap)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)

        # Buat histogram sebelum fuzzy equalization
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.hist(np.array(grayscale_image).ravel(), bins=256,
                 range=(0, 256), density=True, color='b', alpha=0.6)
        plt.title('Histogram Sebelum Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')

        # Buat histogram sesudah fuzzy equalization
        plt.subplot(122)
        fuzzy_equalized_image_flat = np.array(fuzzy_equalized_image).ravel()
        plt.hist(fuzzy_equalized_image_flat, bins=256, range=(
            0, 256), density=True, color='r', alpha=0.6)
        plt.title('Histogram Sesudah Fuzzy Equalization')
        plt.xlabel('Nilai Pixel')
        plt.ylabel('Frekuensi Relatif')

        plt.tight_layout()
        plt.show()

    def invers(self):
        pixmap = self.label.pixmap()
        if pixmap:
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            inverted_image = QtGui.QImage(
                width, height, QtGui.QImage.Format_RGBA8888)

            for y in range(height):
                for x in range(width):
                    pixel = QtGui.QColor(image.pixel(x, y))
                    inverted_color = QtGui.QColor(
                        255 - pixel.red(), 255 - pixel.green(), 255 - pixel.blue(), pixel.alpha())
                    inverted_image.setPixel(x, y, inverted_color.rgba())

            inverted_pixmap = QtGui.QPixmap.fromImage(inverted_image)
            self.label_2.setPixmap(inverted_pixmap)

    def frameAritmatika(self):
        self.window = QtWidgets.QMainWindow()
        self.ui = p()
        self.ui.setupUi(self.window)
        self.window.show()

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1062, 592)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 20, 500, 500))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setText("")
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(540, 20, 500, 500))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setText("")
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1062, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")

        self.menuAritmatika = QtWidgets.QMenu(self.menubar)
        self.menuAritmatika.setObjectName("menuAritmatika")

        self.menuGeometri = QtWidgets.QMenu(self.menubar)
        self.menuGeometri.setObjectName("menuGeometri")
        self.menuScaling = QtWidgets.QMenu(self.menuGeometri)
        self.menuScaling.setObjectName("menuScaling")
        self.menuFlipping = QtWidgets.QMenu(self.menuGeometri)
        self.menuFlipping.setObjectName("menuFlipping")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuColors = QtWidgets.QMenu(self.menubar)
        self.menuColors.setObjectName("menuColors")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColors)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuColors)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuImage_Prossesing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Prossesing.setObjectName("menuImage_Prossesing")
        self.menuKonvolusi = QtWidgets.QMenu(self.menubar)
        self.menuKonvolusi.setObjectName("menuKonvolusi")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menuKonvolusi)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.openImage)

        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        self.actionSave_As.triggered.connect(self.saveAsImage)

        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.actionExit.triggered.connect(self.exitApplication)

        self.actionUniform_Scaling = QtWidgets.QAction(MainWindow)
        self.actionUniform_Scaling.setObjectName("actionUniform_Scaling")
        self.actionUniform_Scaling.triggered.connect(self.uniformScaling)

        self.actionNon_Uniform_Scaling = QtWidgets.QAction(MainWindow)
        self.actionNon_Uniform_Scaling.setObjectName(
            "actionNon_Uniform_Scaling")
        self.actionNon_Uniform_Scaling.triggered.connect(
            self.nonUniformScaling)

        self.actionCroping = QtWidgets.QAction(MainWindow)
        self.actionCroping.setObjectName("actionCroping")

        self.actionHorizontal = QtWidgets.QAction(MainWindow)
        self.actionHorizontal.setObjectName("actionHorizontal")
        self.actionHorizontal.triggered.connect(self.flipHorizontal)

        self.actionVertical = QtWidgets.QAction(MainWindow)
        self.actionVertical.setObjectName("actionVertical")
        self.actionVertical.triggered.connect(self.flipVertical)

        self.actionTranslasi = QtWidgets.QAction(MainWindow)
        self.actionTranslasi.setObjectName("actionTranslasi")
        self.actionTranslasi.triggered.connect(self.translasi)

        self.actionRotasi = QtWidgets.QAction(MainWindow)
        self.actionRotasi.setObjectName("actionRotasi")
        self.actionRotasi.triggered.connect(self.rotasi)

        self.actionHistogram_Input = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Input.setObjectName("actionHistogram_Input")
        self.actionHistogram_Output = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Output.setObjectName("actionHistogram_Output")
        self.actionHistogram_Input_Output = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Input_Output.setObjectName(
            "actionHistogram_Input_Output")

        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.averageGrayscale)

        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLightness.triggered.connect(self.lightnessGrayscale)

        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.luminanceGrayscale)

        self.actionBrightness = QtWidgets.QAction(MainWindow)
        self.actionBrightness.setObjectName("actionBrightness")

        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")

        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionInvers.triggered.connect(self.invers)

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

        self.actionHE = QtWidgets.QAction(MainWindow)
        self.actionHE.setObjectName("actionHE")
        self.actionHE.triggered.connect(self.histogramEqualization)

        self.actionFHE = QtWidgets.QAction(MainWindow)
        self.actionFHE.setObjectName("actionFHE")
        self.actionFHE.triggered.connect(self.fuzzyHERGB)

        self.actionFHE_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFHE_Grayscale.setObjectName("actionFHE_Grayscale")
        self.actionFHE_Grayscale.triggered.connect(self.fuzzyGreyscale)

        self.actionLow_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filter.setObjectName("actionLow_Pass_Filter")
        self.actionHigh_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHigh_Pass_Filter.setObjectName("actionHigh_Pass_Filter")
        self.actionIdentify = QtWidgets.QAction(MainWindow)
        self.actionIdentify.setObjectName("actionIdentify")
        self.actionRobert = QtWidgets.QAction(MainWindow)
        self.actionRobert.setObjectName("actionRobert")
        self.actionSobel = QtWidgets.QAction(MainWindow)
        self.actionSobel.setObjectName("actionSobel")
        self.actionPrewit = QtWidgets.QAction(MainWindow)
        self.actionPrewit.setObjectName("actionPrewit")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionGaussian_Blur_3_x_3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3_x_3.setObjectName(
            "actionGaussian_Blur_3_x_3")
        self.actionGaussian_Blur_5_x_5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5_x_5.setObjectName(
            "actionGaussian_Blur_5_x_5")
        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")

        self.actionAritmatika = QtWidgets.QAction(MainWindow)
        self.actionAritmatika.setObjectName("actionAritmatika")
        self.actionAritmatika.triggered.connect(self.frameAritmatika)

        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionExit)
        self.menuAritmatika.addAction(self.actionAritmatika)
        self.menuScaling.addAction(self.actionUniform_Scaling)
        self.menuScaling.addAction(self.actionNon_Uniform_Scaling)
        self.menuFlipping.addAction(self.actionHorizontal)
        self.menuFlipping.addAction(self.actionVertical)
        self.menuGeometri.addAction(self.menuScaling.menuAction())
        self.menuGeometri.addAction(self.actionCroping)
        self.menuGeometri.addAction(self.menuFlipping.menuAction())
        self.menuGeometri.addAction(self.actionTranslasi)
        self.menuGeometri.addAction(self.actionRotasi)
        self.menuView.addAction(self.actionHistogram_Input)
        self.menuView.addAction(self.actionHistogram_Output)
        self.menuView.addAction(self.actionHistogram_Input_Output)
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuBit_Depth.addAction(self.action1_bit)
        self.menuBit_Depth.addAction(self.action2_bit)
        self.menuBit_Depth.addAction(self.action3_bit)
        self.menuBit_Depth.addAction(self.action4_bit)
        self.menuBit_Depth.addAction(self.action5_bit)
        self.menuBit_Depth.addAction(self.action6_bit)
        self.menuBit_Depth.addAction(self.action7_bit)
        self.menuColors.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColors.addAction(self.actionBrightness)
        self.menuColors.addAction(self.actionContrast)
        self.menuColors.addAction(self.actionInvers)
        self.menuColors.addAction(self.menuBit_Depth.menuAction())
        self.menuImage_Prossesing.addAction(self.actionHE)
        self.menuImage_Prossesing.addAction(self.actionFHE)
        self.menuImage_Prossesing.addAction(self.actionFHE_Grayscale)
        self.menuEdge_Detection.addAction(self.actionRobert)
        self.menuEdge_Detection.addAction(self.actionSobel)
        self.menuEdge_Detection.addAction(self.actionPrewit)
        self.menuKonvolusi.addAction(self.actionLow_Pass_Filter)
        self.menuKonvolusi.addAction(self.actionHigh_Pass_Filter)
        self.menuKonvolusi.addAction(self.actionIdentify)
        self.menuKonvolusi.addAction(self.menuEdge_Detection.menuAction())
        self.menuKonvolusi.addAction(self.actionSharpen)
        self.menuKonvolusi.addAction(self.actionGaussian_Blur_3_x_3)
        self.menuKonvolusi.addAction(self.actionGaussian_Blur_5_x_5)
        self.menuKonvolusi.addAction(self.actionUnsharp_Masking)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAritmatika.menuAction())
        self.menubar.addAction(self.menuGeometri.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuColors.menuAction())
        self.menubar.addAction(self.menuImage_Prossesing.menuAction())
        self.menubar.addAction(self.menuKonvolusi.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuAritmatika.setTitle(_translate("MainWindow", "Aritmatika"))
        self.menuGeometri.setTitle(_translate("MainWindow", "Geometri"))
        self.menuScaling.setTitle(_translate("MainWindow", "Scaling"))
        self.menuFlipping.setTitle(_translate("MainWindow", "Flipping"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuColors.setTitle(_translate("MainWindow", "Colors"))
        self.menuRGB_to_Grayscale.setTitle(
            _translate("MainWindow", "RGB to Grayscale"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuImage_Prossesing.setTitle(
            _translate("MainWindow", "Image Processing"))
        self.menuKonvolusi.setTitle(_translate("MainWindow", "Konvolusi"))
        self.menuEdge_Detection.setTitle(
            _translate("MainWindow", "Edge Detection"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionUniform_Scaling.setText(
            _translate("MainWindow", "Uniform Scaling"))
        self.actionNon_Uniform_Scaling.setText(
            _translate("MainWindow", "Non-Uniform Scaling"))
        self.actionCroping.setText(_translate("MainWindow", "Cropping"))
        self.actionHorizontal.setText(_translate("MainWindow", "Horizontal"))
        self.actionVertical.setText(_translate("MainWindow", "Vertical"))
        self.actionTranslasi.setText(_translate("MainWindow", "Translasi"))
        self.actionRotasi.setText(_translate("MainWindow", "Rotasi"))
        self.actionHistogram_Input.setText(
            _translate("MainWindow", "Histogram Input"))
        self.actionHistogram_Output.setText(
            _translate("MainWindow", "Histogram Output"))
        self.actionHistogram_Input_Output.setText(
            _translate("MainWindow", "Histogram Input Output"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionBrightness.setText(_translate("MainWindow", "Brightness"))
        self.actionContrast.setText(_translate("MainWindow", "Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.action1_bit.setText(_translate("MainWindow", "1 bit"))
        self.action2_bit.setText(_translate("MainWindow", "2 bit"))
        self.action3_bit.setText(_translate("MainWindow", "3 bit"))
        self.action4_bit.setText(_translate("MainWindow", "4 bit"))
        self.action5_bit.setText(_translate("MainWindow", "5 bit"))
        self.action6_bit.setText(_translate("MainWindow", "6 bit"))
        self.action7_bit.setText(_translate("MainWindow", "7 bit"))
        self.actionHE.setText(_translate("MainWindow", "HE"))
        self.actionFHE.setText(_translate("MainWindow", "FHE RGB"))
        self.actionFHE_Grayscale.setText(
            _translate("MainWindow", "FHE Grayscale"))
        self.actionLow_Pass_Filter.setText(
            _translate("MainWindow", "Low Pass Filter"))
        self.actionHigh_Pass_Filter.setText(
            _translate("MainWindow", "High Pass Filter"))
        self.actionIdentify.setText(_translate("MainWindow", "Identify"))
        self.actionRobert.setText(_translate("MainWindow", "Robert"))
        self.actionSobel.setText(_translate("MainWindow", "Sobel"))
        self.actionPrewit.setText(_translate("MainWindow", "Prewit"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionGaussian_Blur_3_x_3.setText(
            _translate("MainWindow", "Gaussian Blur 3 x 3"))
        self.actionGaussian_Blur_5_x_5.setText(
            _translate("MainWindow", "Gaussian Blur 5 x 5"))
        self.actionUnsharp_Masking.setText(
            _translate("MainWindow", "Unsharp Masking"))
        self.actionAritmatika.setText(_translate("MainWindow", "Aritmatika"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
