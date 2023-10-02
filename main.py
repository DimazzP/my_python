import sys
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPixmap, QImage, QColor, qRgb
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QDialog, QInputDialog, QSlider, QVBoxLayout, QLabel
from aritmatika import Ui_Dialog
# import matplotlib.pyplot as plt
from histogram_rgb import HistogramDialog
from brightness import BrightnessDialog
from contrast import ContrastDialog
from functools import partial
from scipy.ndimage import convolve


class Ui_MainWindow(object):

    def unsharp_masking(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Buat kernel Gaussian 5x5
            gaussian_kernel = [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ]

            # Buat gambar baru untuk menyimpan hasil Gaussian Blur
            gaussian_blur_img = QtGui.QImage(
                width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r, g, b = 0, 0, 0
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))
                                weight = gaussian_kernel[i + 2][j + 2]
                                r += pixel_color.red() * weight
                                g += pixel_color.green() * weight
                                b += pixel_color.blue() * weight

                    r //= 256  # Normalisasi hasil konvolusi
                    g //= 256
                    b //= 256

                    # Tetapkan nilai piksel baru ke gambar hasil Gaussian Blur
                    gaussian_blur_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Buat gambar untuk menyimpan hasil unsharp masking
            unsharp_masked_img = QtGui.QImage(
                width, height, QtGui.QImage.Format_RGB32)

            # Hitung perbedaan antara gambar asli dan gambar Gaussian Blur
            for x in range(width):
                for y in range(height):
                    original_color = QtGui.QColor(img.pixel(x, y))
                    blurred_color = QtGui.QColor(gaussian_blur_img.pixel(x, y))

                    r = original_color.red() - blurred_color.red()
                    g = original_color.green() - blurred_color.green()
                    b = original_color.blue() - blurred_color.blue()

                    # Tambahkan perbedaan ke gambar hasil
                    unsharp_masked_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Setel gambar hasil unsharp masking ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(unsharp_masked_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def sharpen(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Buat kernel custom sharpening 3x3
            sharpen_kernel = [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]

            # Buat gambar baru untuk menyimpan hasil custom sharpening
            sharpened_img = QtGui.QImage(
                width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r, g, b = 0, 0, 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))
                                weight = sharpen_kernel[i + 1][j + 1]
                                r += pixel_color.red() * weight
                                g += pixel_color.green() * weight
                                b += pixel_color.blue() * weight

                    r = max(0, min(r, 255))  # Clamp the values to 0-255
                    g = max(0, min(g, 255))
                    b = max(0, min(b, 255))

                    # Setel nilai piksel baru ke gambar hasil
                    sharpened_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Setel gambar hasil custom sharpening ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(sharpened_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def edge_detection_sobel(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Buat kernel Sobel untuk deteksi tepi horizontal
            sobel_kernel_x = [
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ]

            # Buat kernel Sobel untuk deteksi tepi vertikal
            sobel_kernel_y = [
                [1, 2, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ]

            # Buat gambar baru untuk menyimpan hasil deteksi tepi
            edge_img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r_x, g_x, b_x = 0, 0, 0
                    r_y, g_y, b_y = 0, 0, 0
                    r, g, b = 0, 0, 0

                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))
                                weight_x = sobel_kernel_x[i + 1][j + 1]
                                weight_y = sobel_kernel_y[i + 1][j + 1]

                                r_x, g_x, b_x = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                                r_y, g_y, b_y = pixel_color.red(), pixel_color.green(), pixel_color.blue()

                                r = max(0, min(r + (r_x * weight_x), 255))
                                g = max(0, min(g + (g_x * weight_x), 255))
                                b = max(0, min(b + (b_x * weight_x), 255))

                                r = max(0, min(r + (r_y * weight_y), 255))
                                g = max(0, min(g + (g_y * weight_y), 255))
                                b = max(0, min(b + (b_y * weight_y), 255))

                    # Hitung magnitude dari gradien tepi
                    magnitude = int(
                        (r_x**2 + r_y**2 + g_x**2 + g_y**2 + b_x**2 + b_y**2)**0.5)

                    # Clamp nilai magnitude ke dalam rentang 0-255
                    # magnitude = max(0, min(magnitude, 255))

                    # Setel nilai piksel baru ke gambar hasil
                    edge_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Setel gambar hasil deteksi tepi ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(edge_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def edge_detection_prewit(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Kernel Prewitt kombinasi (magnitude)
            prewitt_kernel_x = [
                [-1, -1, -1],
                [0, 0, 0],
                [1, 1, 1]
            ]
            prewitt_kernel_y = [
                [-1, 0, 1],
                [-1, 0, 1],
                [-1, 0, 1]
            ]

            # Buat gambar baru untuk menyimpan hasil deteksi tepi
            edge_img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r, g, b = 0, 0, 0

                    for i in range(3):
                        for j in range(3):
                            px = x + i - 1
                            py = y + j - 1

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))

                                weight_x = prewitt_kernel_x[i][j]
                                weight_y = prewitt_kernel_y[i][j]

                                r_x, g_x, b_x = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                                r_y, g_y, b_y = pixel_color.red(), pixel_color.green(), pixel_color.blue()

                                r = max(0, min(r + (r_x * weight_x), 255))
                                g = max(0, min(g + (g_x * weight_x), 255))
                                b = max(0, min(b + (b_x * weight_x), 255))

                                r = max(0, min(r + (r_y * weight_y), 255))
                                g = max(0, min(g + (g_y * weight_y), 255))
                                b = max(0, min(b + (b_y * weight_y), 255))

                    # Clamp nilai warna ke dalam rentang 0-255
                    magnitude = int(r ^ 2 + g ^ 2 + b ^ 2)

                    # Clamp nilai magnitude ke dalam rentang 0-255
                    magnitude1 = max(0, min(magnitude, 255))
                    # Setel nilai piksel baru ke gambar hasil
                    edge_img.setPixel(x, y, QtGui.qRgb(
                        magnitude1, magnitude1, magnitude1))

            # Setel gambar hasil deteksi tepi ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(edge_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def edge_detection_robert(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Kernel Robert untuk Gradien Horizontal (Gx)
            robert_kernel_x = [
                [-1, 0],
                [0, 1]
            ]

            # Kernel Robert untuk Gradien Vertikal (Gy)
            robert_kernel_y = [
                [0, -1],
                [1, 0]
            ]

            # Buat gambar baru untuk menyimpan hasil deteksi tepi
            edge_img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r_x, g_x, b_x = 0, 0, 0
                    r_y, g_y, b_y = 0, 0, 0

                    for i in range(2):
                        for j in range(2):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))

                                weight_x = robert_kernel_x[i][j]
                                weight_y = robert_kernel_y[i][j]

                                r_x += pixel_color.red() * weight_x
                                g_x += pixel_color.green() * weight_x
                                b_x += pixel_color.blue() * weight_x

                                r_y += pixel_color.red() * weight_y
                                g_y += pixel_color.green() * weight_y
                                b_y += pixel_color.blue() * weight_y

                    # Hitung magnitude dari gradien tepi
                    magnitude = int(
                        (r_x**2 + r_y**2 + g_x**2 + g_y**2 + b_x**2 + b_y**2)**0.5)

                    # Clamp nilai magnitude ke dalam rentang 0-255
                    magnitude = max(0, min(magnitude, 255))

                    # Setel nilai piksel baru ke gambar hasil
                    edge_img.setPixel(x, y, QtGui.qRgb(
                        magnitude, magnitude, magnitude))

            # Setel gambar hasil deteksi tepi ke label atau tempat yang sesuai
            output_pixmap = QPixmap.fromImage(edge_img)
            self.label_gambar_tujuan.setPixmap(output_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def identity(self):
        pixmap = self.label_gambar_asal.pixmap()

        if pixmap:
            img = pixmap.toImage()

            # Buat salinan gambar asli ke gambar tujuan
            output_pixmap = QPixmap.fromImage(img)

            self.label_gambar_tujuan.setPixmap(output_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def gaussian3x3(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Buat kernel Gaussian 3x3
            kernel = [
                [1, 2, 1],
                [2, 4, 2],
                [1, 2, 1]
            ]

            # Buat gambar baru untuk menyimpan hasil Gaussian Blur
            new_img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r, g, b = 0, 0, 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))
                                weight = kernel[i + 1][j + 1]
                                r += pixel_color.red() * weight
                                g += pixel_color.green() * weight
                                b += pixel_color.blue() * weight

                    r //= 16  # Normalisasi hasil konvolusi
                    g //= 16
                    b //= 16

                    # Tetapkan nilai piksel baru ke gambar hasil
                    new_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Setel gambar hasil ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(new_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def gaussian5x5(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Buat kernel Gaussian 5x5
            kernel = [
                [1, 4, 6, 4, 1],
                [4, 16, 24, 16, 4],
                [6, 24, 36, 24, 6],
                [4, 16, 24, 16, 4],
                [1, 4, 6, 4, 1]
            ]

            # Buat gambar baru untuk menyimpan hasil Gaussian Blur
            new_img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r, g, b = 0, 0, 0
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))
                                weight = kernel[i + 2][j + 2]
                                r += pixel_color.red() * weight
                                g += pixel_color.green() * weight
                                b += pixel_color.blue() * weight

                    r //= 256  # Normalisasi hasil konvolusi
                    g //= 256
                    b //= 256

                    # Tetapkan nilai piksel baru ke gambar hasil
                    new_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Setel gambar hasil ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(new_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def high_pass_filter(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            high_pass_kernel = np.array([
                [0, -1, 0],
                [-1, 4, -1],
                [0, -1, 0]
            ], dtype=np.float32)

        # Buat gambar baru untuk menyimpan hasil filter tinggi
        high_pass_img = QImage(width, height, QImage.Format_RGB32)

        for x in range(width):
            for y in range(height):
                r, g, b = 0, 0, 0

                for i in range(-1, 2):
                    for j in range(-1, 2):
                        px = x + i
                        py = y + j

                        if 0 <= px < width and 0 <= py < height:
                            pixel_color = QColor(img.pixel(px, py))
                            r += pixel_color.red() * \
                                high_pass_kernel[i + 1][j + 1]
                            g += pixel_color.green() * \
                                high_pass_kernel[i + 1][j + 1]
                            b += pixel_color.blue() * \
                                high_pass_kernel[i + 1][j + 1]

                # Clamp nilai warna ke dalam rentang 0-255
                r = max(0, min(int(r), 255))
                g = max(0, min(int(g), 255))
                b = max(0, min(int(b), 255))

                # Setel nilai piksel baru ke gambar hasil
                high_pass_img.setPixel(x, y, QColor(r, g, b).rgb())

        # Tampilkan hasil filter tinggi di label atau tempat yang sesuai
        high_pass_pixmap = QPixmap.fromImage(high_pass_img)
        self.label_gambar_tujuan.setPixmap(high_pass_pixmap)
        self.label_gambar_tujuan.setScaledContents(True)

    def low_pass_filter(self):
        pixmap = self.label_gambar_asal.pixmap()
        if pixmap:
            img = pixmap.toImage()
            width = img.width()
            height = img.height()

            # Buat kernel Gaussian 3x3
            kernel = [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]

            # Buat gambar baru untuk menyimpan hasil Gaussian Blur
            new_img = QtGui.QImage(width, height, QtGui.QImage.Format_RGB32)

            for x in range(width):
                for y in range(height):
                    r, g, b = 0, 0, 0
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            px = x + i
                            py = y + j

                            if 0 <= px < width and 0 <= py < height:
                                pixel_color = QtGui.QColor(img.pixel(px, py))
                                weight = kernel[i + 1][j + 1]
                                r += pixel_color.red() * weight
                                g += pixel_color.green() * weight
                                b += pixel_color.blue() * weight

                    r //= 9  # Normalisasi hasil konvolusi
                    g //= 9
                    b //= 9

                    # Tetapkan nilai piksel baru ke gambar hasil
                    new_img.setPixel(x, y, QtGui.qRgb(r, g, b))

            # Setel gambar hasil ke label atau tempat yang sesuai
            ouput_pixmap = QPixmap.fromImage(new_img)
            self.label_gambar_tujuan.setPixmap(ouput_pixmap)
            self.label_gambar_tujuan.setScaledContents(True)

    def fhe_grayscale(self):
        alpha = 1
        pixmap = self.label_gambar_asal.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()

        # Inisialisasi histogram untuk gambar grayscale
        grayscale_histogram = [0] * 256

        # Hitung histogram untuk gambar grayscale
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_histogram[grayscale_value] += 1

        # Hitung cumulative histogram untuk gambar grayscale
        cumulative_histogram = self.calculate_cumulative_histogram(
            grayscale_histogram)

        # Terapkan fuzzy histogram equalization pada gambar grayscale
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                fuzzy_grayscale = alpha * \
                    cumulative_histogram[grayscale_value] + \
                    (1 - alpha) * grayscale_value
                new_color = QColor(
                    fuzzy_grayscale, fuzzy_grayscale, fuzzy_grayscale)
                image.setPixel(x, y, new_color.rgb())  # Set warna piksel baru

        pixmap_equalized = QPixmap.fromImage(image)
        self.label_gambar_tujuan.setPixmap(pixmap_equalized)
        self.label_gambar_tujuan.setScaledContents(True)

    def fhe_rgb(self):
        alpha = 1  # Mengatur alpha menjadi 0
        pixmap = self.label_gambar_asal.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()

        # Inisialisasi histogram untuk setiap saluran
        red_histogram = [0] * 256
        green_histogram = [0] * 256
        blue_histogram = [0] * 256

        # Hitung histogram untuk setiap saluran
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                red_histogram[pixel_color.red()] += 1
                green_histogram[pixel_color.green()] += 1
                blue_histogram[pixel_color.blue()] += 1

        # Hitung cumulative histogram untuk setiap saluran
        red_cumulative_histogram = self.calculate_cumulative_histogram(
            red_histogram)
        green_cumulative_histogram = self.calculate_cumulative_histogram(
            green_histogram)
        blue_cumulative_histogram = self.calculate_cumulative_histogram(
            blue_histogram)

        # Terapkan fuzzy histogram equalization pada setiap saluran
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r = pixel_color.red()
                g = pixel_color.green()
                b = pixel_color.blue()
                fuzzy_r = alpha * red_cumulative_histogram[r] + (1 - alpha) * r
                fuzzy_g = alpha * \
                    green_cumulative_histogram[g] + (1 - alpha) * g
                fuzzy_b = alpha * \
                    blue_cumulative_histogram[b] + (1 - alpha) * b
                new_color = QColor(int(fuzzy_r), int(fuzzy_g), int(fuzzy_b))
                image.setPixelColor(x, y, new_color)

        pixmap_equalized = QPixmap.fromImage(image)
        self.label_gambar_tujuan.setPixmap(pixmap_equalized)
        self.label_gambar_tujuan.setScaledContents(True)

    def histogram_equalization(self):
        pixmap = self.label_gambar_asal.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()

        # Inisialisasi histogram untuk setiap saluran
        red_histogram = [0] * 256
        green_histogram = [0] * 256
        blue_histogram = [0] * 256

        # Hitung histogram untuk setiap saluran
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                red_histogram[pixel_color.red()] += 1
                green_histogram[pixel_color.green()] += 1
                blue_histogram[pixel_color.blue()] += 1

        # Hitung cumulative histogram untuk setiap saluran
        red_cumulative_histogram = self.calculate_cumulative_histogram(
            red_histogram)
        green_cumulative_histogram = self.calculate_cumulative_histogram(
            green_histogram)
        blue_cumulative_histogram = self.calculate_cumulative_histogram(
            blue_histogram)

        # Terapkan histogram equalization pada setiap saluran
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r = red_cumulative_histogram[pixel_color.red()]
                g = green_cumulative_histogram[pixel_color.green()]
                b = blue_cumulative_histogram[pixel_color.blue()]
                new_color = QColor(r, g, b)
                image.setPixelColor(x, y, new_color)

        pixmap_equalized = QPixmap.fromImage(image)
        self.label_gambar_tujuan.setPixmap(pixmap_equalized)
        self.label_gambar_tujuan.setScaledContents(True)

    def calculate_cumulative_histogram(self, histogram):
        cumulative_histogram = [0] * 256
        cumulative_sum = 0
        total_pixels = sum(histogram)

        for i in range(256):
            cumulative_sum += histogram[i]
            cumulative_histogram[i] = int(255 * cumulative_sum / total_pixels)

        return cumulative_histogram

    def threshold(self):
        print('test')
        threshold_value, ok = QtWidgets.QInputDialog.getInt(
            None, "threshold", "Masukkan nilai threshold 0-255:")
        if ok:
            pixmap = self.label_gambar_asal.pixmap()
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            for y in range(height):
                for x in range(width):
                    pixel_color = image.pixelColor(x, y)
                    r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                    grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                    if grayscale_value > threshold_value:
                        new_color = QColor(255, 255, 255)  # Putih
                    else:
                        new_color = QColor(0, 0, 0)  # Hitam
                    image.setPixelColor(x, y, new_color)

            pixmap_thresholded = QPixmap.fromImage(image)
            self.label_gambar_tujuan.setPixmap(pixmap_thresholded)
            self.label_gambar_tujuan.setScaledContents(True)

    def search_bit(self, my_value, bit):
        bit_value = 2 ** bit
        searchThreshold = int(256 / bit_value)
        searchValue = int(256 / (bit_value-1))
        threshold = []
        check_value = []
        result = None
        threshold.append(0)
        check_value.append(0)
        for j in range(1, bit_value):
            arr_value = j * searchValue
            check_value.append(arr_value)
        for i in range(1, bit_value + 1):
            value = i * searchThreshold
            threshold.append(value)
        for index, t in enumerate(threshold):
            if not my_value >= t:
                check_index = index - 1
                result = check_value[check_index]
                break
        return int(result)

    def bit_depth(self, bit):
        pixmap = self.label_gambar_asal.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        self.search_bit(128, bit)
        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                r = min(max(self.search_bit(r, bit), 0), 255)
                g = min(max(self.search_bit(g, bit), 0), 255)
                b = min(max(self.search_bit(b, bit), 0), 255)
                new_color = QColor(r, g, b)
                image.setPixelColor(x, y, new_color)

        new_image = QPixmap.fromImage(image)
        self.label_gambar_tujuan.setPixmap(new_image)
        self.label_gambar_tujuan.setScaledContents(True)

    def invers(self):
        pixmap = self.label_gambar_asal.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()

        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                r = min(max(255 - r, 0), 255)
                g = min(max(255 - g, 0), 255)
                b = min(max(255 - b, 0), 255)
                new_color = QColor(r, g, b)
                image.setPixelColor(x, y, new_color)

        pixmap_grayscale = QPixmap.fromImage(image)
        self.label_gambar_tujuan.setPixmap(pixmap_grayscale)
        self.label_gambar_tujuan.setScaledContents(True)

    def brightnes(self):
        if self.label_gambar_asal.pixmap() is not None:
            self.brightnes_dialog = BrightnessDialog(main_window=self)
            self.brightnes_dialog.exec_()
        else:
            QMessageBox.warning(
                MainWindow, "Peringatan", "Tidak ada gambar yang dibuka.")

    def contrast(self):
        if self.label_gambar_asal.pixmap() is not None:
            self.contrast_dialog = ContrastDialog(main_window=self)
            self.contrast_dialog.exec_()
        else:
            QMessageBox.warning(
                MainWindow, "Peringatan", "Tidak ada gambar yang dibuka.")

    def histogram_input(self):
        self.histogram_input_dialog = HistogramDialog(
            self.directory_input)
        self.histogram_input_dialog.show()

    def histogram_output(self):
        if hasattr(self, 'directory_input'):
            output_file = "output.jpg"  # Nama file output yang akan digunakan
            pixmap = self.label_gambar_tujuan.pixmap()

            if pixmap:
                # Mengambil QImage dari pixmap
                image = pixmap.toImage()

                # Simpan QImage sebagai file jpg
                if image.save(output_file, "jpg"):
                    self.histogram_output_dialog = HistogramDialog(output_file)
                    self.histogram_output_dialog.show()
                else:
                    QtWidgets.QMessageBox.critical(
                        None, "Error", "Gagal menyimpan gambar.")
            else:
                QtWidgets.QMessageBox.critical(
                    None, "Error", "Tidak ada gambar yang dimuat.")
        else:
            QtWidgets.QMessageBox.critical(
                None, "Error", "Tidak ada gambar yang dimuat.")

    def histogram_input_output(self):
        self.histogram_input()
        self.histogram_output()

    def translasi(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            # Meminta pengguna memasukkan nilai tx (geser horizontal)
            tx, ok_tx = QtWidgets.QInputDialog.getInt(
                None, "Translate Image", "Masukkan nilai tx (geser horizontal):")

            if ok_tx:
                # Meminta pengguna memasukkan nilai ty (geser vertikal)
                ty, ok_ty = QtWidgets.QInputDialog.getInt(
                    None, "Translate Image", "Masukkan nilai ty (geser vertikal):")

                if ok_ty:
                    width = original_pixmap.width()
                    height = original_pixmap.height()
                    # Membuat QPixmap baru dengan ukuran yang sama
                    translated_pixmap = QtGui.QPixmap(width, height)
                    # Mengisi dengan latar belakang transparan
                    translated_pixmap.fill(QtGui.QColor(0, 0, 0, 0))
                    painter = QtGui.QPainter(translated_pixmap)
                    for x in range(width):
                        for y in range(height):
                            # Menggeser koordinat x dan y sesuai dengan nilai tx dan ty
                            x_translated = x + tx
                            y_translated = y + ty
                            # Memeriksa apakah koordinat baru berada dalam batas gambar
                            if 0 <= x_translated < width and 0 <= y_translated < height:
                                # Mendapatkan warna pixel dari gambar asli
                                pixel_color = original_pixmap.toImage().pixelColor(x, y)
                                # Menggambar ulang pixel ke gambar yang sudah digeser
                                painter.setPen(pixel_color)
                                painter.drawPoint(x_translated, y_translated)

                    painter.end()  # Mengakhiri proses menggambar

                    self.label_gambar_tujuan.setScaledContents(True)
                    self.label_gambar_tujuan.setPixmap(translated_pixmap)
                    self.label_gambar_tujuan.setAlignment(
                        QtCore.Qt.AlignCenter)
                else:
                    QtWidgets.QMessageBox.warning(
                        None, "Error", "Masukkan nilai ty yang valid.")
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan nilai tx yang valid.")

    # todo rotasi
    def rotasi(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            # Menghitung pusat rotasi
            center_x = original_pixmap.width() / 2
            center_y = original_pixmap.height() / 2

            # Meminta pengguna memasukkan sudut rotasi
            angle, ok = QtWidgets.QInputDialog.getInt(
                None, "Rotate Image", "Masukkan sudut rotasi (derajat):")

            if ok:
                # Membuat QPixmap baru dengan ukuran yang sama
                rotated_pixmap = QtGui.QPixmap(original_pixmap.size())
                # Mengisi dengan latar belakang transparan
                rotated_pixmap.fill(QtGui.QColor(0, 0, 0, 0))

                # Membuat transformasi rotasi
                transform = QtGui.QTransform()
                transform.translate(center_x, center_y)
                transform.rotate(angle)
                transform.translate(-center_x, -center_y)

                # Melakukan rotasi gambar
                painter = QtGui.QPainter(rotated_pixmap)
                painter.setTransform(transform)
                painter.drawPixmap(0, 0, original_pixmap)
                painter.end()

                self.label_gambar_tujuan.setScaledContents(True)
                self.label_gambar_tujuan.setPixmap(rotated_pixmap)
                self.label_gambar_tujuan.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan sudut rotasi yang valid.")

    # todo flipping horizontal

    def flipHorizontal(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            width = original_pixmap.width()
            height = original_pixmap.height()

            # Membuat QPixmap baru dengan ukuran yang sama
            flipped_pixmap = QtGui.QPixmap(width, height)
            # Mengisi dengan latar belakang transparan
            flipped_pixmap.fill(QtGui.QColor(0, 0, 0, 0))

            painter = QtGui.QPainter(flipped_pixmap)

            for x in range(width):
                for y in range(height):
                    # Menghitung koordinat x yang diflip
                    x_flipped = width - 1 - x

                    # Mendapatkan warna pixel dari gambar asli
                    pixel_color = original_pixmap.toImage().pixelColor(x, y)

                    # Menggambar ulang pixel ke gambar yang sudah diflip
                    painter.setPen(pixel_color)
                    painter.drawPoint(x_flipped, y)

            painter.end()  # Mengakhiri proses menggambar

            self.label_gambar_tujuan.setScaledContents(True)
            self.label_gambar_tujuan.setPixmap(flipped_pixmap)
            self.label_gambar_tujuan.setAlignment(QtCore.Qt.AlignCenter)

    # todo flipping vertical
    def flipVertical(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            width = original_pixmap.width()
            height = original_pixmap.height()

            # Membuat QPixmap baru dengan ukuran yang sama
            flipped_pixmap = QtGui.QPixmap(width, height)
            # Mengisi dengan latar belakang transparan
            flipped_pixmap.fill(QtGui.QColor(0, 0, 0, 0))

            painter = QtGui.QPainter(flipped_pixmap)

            for x in range(width):
                for y in range(height):
                    # Membalik koordinat y
                    y_flipped = height - 1 - y

                    # Menyimpan nilai x seperti semula
                    x_flipped = x

                    # Mendapatkan warna pixel dari gambar asli
                    pixel_color = original_pixmap.toImage().pixelColor(x, y)

                    # Menggambar ulang pixel ke gambar yang sudah diflip
                    painter.setPen(pixel_color)
                    painter.drawPoint(x_flipped, y_flipped)

            painter.end()  # Mengakhiri proses menggambar

            self.label_gambar_tujuan.setScaledContents(False)
            self.label_gambar_tujuan.setPixmap(flipped_pixmap)
            self.label_gambar_tujuan.setAlignment(QtCore.Qt.AlignCenter)

    def flipping(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            flip_horizontal, ok_horizontal = QtWidgets.QInputDialog.getInt(
                None, "Flip Image", "1. Flip Horizontal\n2. Flip Vertical\nPilih jenis flipping:")

            if ok_horizontal:
                transformed_pixmap = None
                if flip_horizontal == 1:
                    # Flip horizontal
                    transform = QtGui.QTransform()
                    transform.scale(-1, 1)
                    transformed_pixmap = original_pixmap.transformed(transform)
                elif flip_horizontal == 2:
                    # Flip vertical
                    transform = QtGui.QTransform()
                    transform.scale(1, -1)
                    transformed_pixmap = original_pixmap.transformed(transform)

                if transformed_pixmap:
                    self.label_gambar_tujuan.setScaledContents(False)
                    self.label_gambar_tujuan.setPixmap(transformed_pixmap)
                    self.label_gambar_tujuan.setAlignment(
                        QtCore.Qt.AlignCenter)
                else:
                    QtWidgets.QMessageBox.warning(
                        None, "Error", "Terjadi kesalahan dalam flipping gambar.")
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Pilih jenis flipping yang valid (1 atau 2).")

    def cropping(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            x, ok_x = QtWidgets.QInputDialog.getInt(
                None, "Crop Image", "Masukkan koordinat X:")
            y, ok_y = QtWidgets.QInputDialog.getInt(
                None, "Crop Image", "Masukkan koordinat Y:")
            width, ok_width = QtWidgets.QInputDialog.getInt(
                None, "Crop Image", "Masukkan lebar:")
            height, ok_height = QtWidgets.QInputDialog.getInt(
                None, "Crop Image", "Masukkan tinggi:")

            if ok_x and ok_y and ok_width and ok_height:
                cropped_pixmap = original_pixmap.copy(x, y, width, height)
                self.label_gambar_tujuan.setScaledContents(False)
                self.label_gambar_tujuan.setPixmap(cropped_pixmap)
                self.label_gambar_tujuan.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan nilai yang valid untuk X, Y, lebar, dan tinggi.")

    # todo scaling uniform

    def scalingUniform(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            scale_factor, _ = QtWidgets.QInputDialog.getDouble(
                None, "Uniform Scaling", "Masukkan skala:")
            if scale_factor > 0:
                scaled_pixmap = original_pixmap.scaled(
                    original_pixmap.size() * scale_factor, QtCore.Qt.KeepAspectRatio)
                self.label_gambar_tujuan.setScaledContents(False)
                self.label_gambar_tujuan.setPixmap(scaled_pixmap)
                self.label_gambar_tujuan.setAlignment(QtCore.Qt.AlignCenter)
            else:
                QtWidgets.QMessageBox.warning(
                    None, "Error", "Masukkan bilangan positif.")

    # todo scaling non uniform

    def scalingNonUniform(self):
        original_pixmap = self.label_gambar_asal.pixmap()
        if original_pixmap:
            scale_factor_x, _ = QtWidgets.QInputDialog.getDouble(
                None, "Non-Uniform Scaling", "Masukkan skala-X:")
            scale_factor_y, _ = QtWidgets.QInputDialog.getDouble(
                None, "Non-Uniform Scaling", "Masukkan skala-Y")

        if scale_factor_x > 0 and scale_factor_y > 0:
            d = QtGui.Qd()
            d.scale(scale_factor_x, scale_factor_y)
            scaled_pixmap = original_pixmap.ded(d)
            self.label_gambar_tujuan.setScaledContents(False)
            self.label_gambar_tujuan.setPixmap(
                QtGui.QPixmap.fromImage(scaled_pixmap.toImage()))
            self.label_gambar_tujuan.setAlignment(QtCore.Qt.AlignCenter)
        else:
            QtWidgets.QMessageBox.warning(
                None, "Error", "Masukkan bilangan positif.")

    # todo aritmetical operation
    def open_aritmatical(self):
        self.dialog_aritmatika = QDialog()
        self.ui_aritmatika = Ui_Dialog()
        self.ui_aritmatika.setupUi(self.dialog_aritmatika)
        self.dialog_aritmatika.show()

    # todo ubah ke grayscale luminance
    def greyscale_luminance(self):
        pixmap = self.label_gambar_asal.pixmap()
        image = pixmap.toImage()
        width = image.width()
        height = image.height()
        grayscale_image = QImage(width, height, QImage.Format_RGB888)

        for y in range(height):
            for x in range(width):
                pixel_color = image.pixelColor(x, y)
                r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()

                gray_value = int(0.299 * r + 0.587 * g + 0.114 * b)
                grayscale_color = QColor(gray_value, gray_value, gray_value)
                grayscale_image.setPixelColor(x, y, grayscale_color)

        pixmap_grayscale = QPixmap.fromImage(grayscale_image)
        self.label_gambar_tujuan.setPixmap(pixmap_grayscale)
        self.label_gambar_tujuan.setScaledContents(True)
        print('sudah luminance')

    # todo ubah ke grayscale lightnes
    def greyscale_lightness(self):
        pixmap = self.label_gambar_asal.pixmap()
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

        # Tampilkan gambar grayscale di label_gambar_tujuan
        pixmap_grayscale = QPixmap.fromImage(grayscale_image)
        self.label_gambar_tujuan.setPixmap(pixmap_grayscale)
        self.label_gambar_tujuan.setScaledContents(True)
        print('sudah lightnes')

    # todo ubah ke grayscale average
    def grayscale_average(self):
        pixmap = self.label_gambar_asal.pixmap()
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

        # Tampilkan gambar grayscale di label_gambar_tujuan
        pixmap_grayscale = QPixmap.fromImage(grayscale_image)
        self.label_gambar_tujuan.setPixmap(pixmap_grayscale)
        self.label_gambar_tujuan.setScaledContents(True)
        print('sudah average')

    # todo keluar

    def keluar_aplikasi(self):
        sys.exit(app.exec_())

    # todo buka file
    def buka_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getOpenFileName(
            MainWindow, "Buka File Gambar", "", "Gambar (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            self.directory_input = file_name
            pixmap = QPixmap(file_name)
            self.label_gambar_asal.setPixmap(pixmap)
            self.label_gambar_asal.setScaledContents(True)

    # todo simpan sebagai
    def simpan_sebagai(self):
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        file_name, _ = QFileDialog.getSaveFileName(
            MainWindow, "Simpan Gambar Sebagai", "", "Gambar (*.png *.jpg *.jpeg *.bmp *.gif)", options=options)
        if file_name:
            pixmap = self.label_gambar_tujuan.pixmap()
            if pixmap:
                if pixmap.save(file_name):
                    QMessageBox.information(
                        MainWindow, "Sukses", "Gambar telah disimpan dengan nama yang berbeda.")
                else:
                    QMessageBox.critical(
                        MainWindow, "Gagal", "Gagal menyimpan gambar.")
            else:
                QMessageBox.warning(
                    MainWindow, "Peringatan", "Tidak ada gambar yang ditampilkan untuk disimpan.")

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1000, 650)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.directory_input = "/path/to/your/default/directory"
        self.directory_output = "/path/to/your/default/directory"
        self.histogram_input_dialog = None  # Inisialisasi dialog
        self.histogram_output_dialog = None  # Inisialisasi dialog
        self.brightnes_dialog = None  # Inisialisasi dialog
        self.contrast_dialog = None  # Inisialisasi dialog
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(20, 60, 460, 460))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.verticalLayoutWidget.setStyleSheet("border: 1px solid black;")
        self.label_gambar_asal = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_gambar_asal.setText("")
        self.label_gambar_asal.setObjectName("label_gambar_asal")
        self.verticalLayout.addWidget(self.label_gambar_asal)
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(
            QtCore.QRect(520, 60, 460, 460))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(
            self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayoutWidget_2.setStyleSheet("border: 1px solid black;")
        self.label_gambar_tujuan = QtWidgets.QLabel(
            self.verticalLayoutWidget_2)
        self.label_gambar_tujuan.setText("")
        self.label_gambar_tujuan.setObjectName("label_gambar_tujuan")
        self.verticalLayout_2.addWidget(self.label_gambar_tujuan)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 878, 21))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuView = QtWidgets.QMenu(self.menubar)
        self.menuView.setObjectName("menuView")
        self.menuHistogram = QtWidgets.QMenu(self.menuView)
        self.menuHistogram.setObjectName("menuHistogram")
        self.menuColor = QtWidgets.QMenu(self.menubar)
        self.menuColor.setObjectName("menuColor")
        self.menuRGB = QtWidgets.QMenu(self.menuColor)
        self.menuRGB.setObjectName("menuRGB")
        self.menuRGB_to_Grayscale = QtWidgets.QMenu(self.menuColor)
        self.menuRGB_to_Grayscale.setObjectName("menuRGB_to_Grayscale")
        self.actionBrightness = QtWidgets.QAction(self.menuColor)
        self.actionBrightness.setObjectName("actionBrightness")
        self.menuBit_Depth = QtWidgets.QMenu(self.menuColor)
        self.menuBit_Depth.setObjectName("menuBit_Depth")
        self.menuTentang = QtWidgets.QMenu(self.menubar)
        self.menuTentang.setObjectName("menuTentang")
        self.menuImage_Processing = QtWidgets.QMenu(self.menubar)
        self.menuImage_Processing.setObjectName("menuImage_Processing")
        self.menuAritmetical_Operation = QtWidgets.QMenu(self.menubar)
        self.menuAritmetical_Operation.setObjectName(
            "menuAritmetical_Operation")
        self.menuIdentity = QtWidgets.QMenu(self.menubar)
        self.menuIdentity.setObjectName("menuIdentity")
        self.menuEdge_Detection = QtWidgets.QMenu(self.menuIdentity)
        self.menuEdge_Detection.setObjectName("menuEdge_Detection")
        self.menuGaussian_Blur = QtWidgets.QMenu(self.menuIdentity)
        self.menuGaussian_Blur.setObjectName("menuGaussian_Blur")
        self.menuEdge_Detection_2 = QtWidgets.QMenu(self.menubar)
        self.menuEdge_Detection_2.setObjectName("menuEdge_Detection_2")
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
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.simpanSebagai = QtWidgets.QAction(MainWindow)
        self.simpanSebagai.setObjectName("simpanSebagai")
        self.actionKeluar = QtWidgets.QAction(MainWindow)
        self.actionKeluar.setObjectName("actionKeluar")
        self.actionInvers = QtWidgets.QAction(MainWindow)
        self.actionInvers.setObjectName("actionInvers")
        self.actionthreshold = QtWidgets.QAction(MainWindow)
        self.actionthreshold.setObjectName("actionthreshold")
        self.actionLog_Brightness = QtWidgets.QAction(MainWindow)
        self.actionLog_Brightness.setObjectName("actionLog_Brightness")
        self.actionGamma_Correction = QtWidgets.QAction(MainWindow)
        self.actionGamma_Correction.setObjectName("actionGamma_Correction")
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName(
            "actionHistogram_Equalization")
        self.actionFuzzy_HE_RGB = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_HE_RGB.setObjectName("actionFuzzy_HE_RGB")
        self.actionFuzzy_Grayscale = QtWidgets.QAction(MainWindow)
        self.actionFuzzy_Grayscale.setObjectName("actionFuzzy_Grayscale")
        self.actionSharpen = QtWidgets.QAction(MainWindow)
        self.actionSharpen.setObjectName("actionSharpen")
        self.actionUnsharp_Masking = QtWidgets.QAction(MainWindow)
        self.actionUnsharp_Masking.setObjectName("actionUnsharp_Masking")
        self.actionIdentity = QtWidgets.QAction(MainWindow)
        self.actionIdentity.setObjectName("actionIdentity")
        self.actionLow_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionLow_Pass_Filter.setObjectName("actionLow_Pass_Filter")
        self.actionHigh_Pass_Filter = QtWidgets.QAction(MainWindow)
        self.actionHigh_Pass_Filter.setObjectName("actionHigh_Pass_Filter")
        self.actionBandstop_Filter = QtWidgets.QAction(MainWindow)
        self.actionBandstop_Filter.setObjectName("actionBandstop_Filter")
        self.actionKuning = QtWidgets.QAction(MainWindow)
        self.actionKuning.setObjectName("actionKuning")
        self.actionOrange = QtWidgets.QAction(MainWindow)
        self.actionOrange.setObjectName("actionOrange")
        self.actionCyan = QtWidgets.QAction(MainWindow)
        self.actionCyan.setObjectName("actionCyan")
        self.actionPurple = QtWidgets.QAction(MainWindow)
        self.actionPurple.setObjectName("actionPurple")
        self.actionGrey = QtWidgets.QAction(MainWindow)
        self.actionGrey.setObjectName("actionGrey")
        self.actionCoklat = QtWidgets.QAction(MainWindow)
        self.actionCoklat.setObjectName("actionCoklat")
        self.actionMerah = QtWidgets.QAction(MainWindow)
        self.actionMerah.setObjectName("actionMerah")
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionLightness = QtWidgets.QAction(MainWindow)
        self.actionLightness.setObjectName("actionLightness")
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionFlippingHorizontal = QtWidgets.QAction(MainWindow)
        self.actionFlippingHorizontal.setObjectName("actionFlippingHorizontal")
        self.actionFlippingVertical = QtWidgets.QAction(MainWindow)
        self.actionFlippingVertical.setObjectName("actionFlippingVertical")
        self.actionContrast = QtWidgets.QAction(MainWindow)
        self.actionContrast.setObjectName("actionContrast")
        self.action1_Bit = QtWidgets.QAction(MainWindow)
        self.action1_Bit.setObjectName("action1_Bit")
        self.action2_Bit = QtWidgets.QAction(MainWindow)
        self.action2_Bit.setObjectName("action2_Bit")
        self.action3_Bit = QtWidgets.QAction(MainWindow)
        self.action3_Bit.setObjectName("action3_Bit")
        self.action4_Bit = QtWidgets.QAction(MainWindow)
        self.action4_Bit.setObjectName("action4_Bit")
        self.action5_Bit = QtWidgets.QAction(MainWindow)
        self.action5_Bit.setObjectName("action5_Bit")
        self.action6_Bit = QtWidgets.QAction(MainWindow)
        self.action6_Bit.setObjectName("action6_Bit")
        self.action7_Bit = QtWidgets.QAction(MainWindow)
        self.action7_Bit.setObjectName("action7_Bit")
        self.actionInput = QtWidgets.QAction(MainWindow)
        self.actionInput.setObjectName("actionInput")
        self.actionOutput = QtWidgets.QAction(MainWindow)
        self.actionOutput.setObjectName("actionOutput")
        self.actionInput_Output = QtWidgets.QAction(MainWindow)
        self.actionInput_Output.setObjectName("actionInput_Output")
        self.actionEdge_Detection_Robert = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_Robert.setObjectName(
            "actionEdge_Detection_Robert")
        self.actionEdge_Detection_Sobel = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_Sobel.setObjectName(
            "actionEdge_Detection_Sobel")
        self.actionEdge_Detection_Prewit = QtWidgets.QAction(MainWindow)
        self.actionEdge_Detection_Prewit.setObjectName(
            "actionEdge_Detection_Prewit")
        self.actionGaussian_Blur_3x3 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_3x3.setObjectName("actionGaussian_Blur_3x3")
        self.actionGaussian_Blur_5_5 = QtWidgets.QAction(MainWindow)
        self.actionGaussian_Blur_5_5.setObjectName("actionGaussian_Blur_5_5")
        self.actionPrewitt = QtWidgets.QAction(MainWindow)
        self.actionPrewitt.setObjectName("actionPrewitt")
        self.actionSquare_3 = QtWidgets.QAction(MainWindow)
        self.actionSquare_3.setObjectName("actionSquare_3")
        self.actionSquare_5 = QtWidgets.QAction(MainWindow)
        self.actionSquare_5.setObjectName("actionSquare_5")
        self.actionCros = QtWidgets.QAction(MainWindow)
        self.actionCros.setObjectName("actionCros")
        self.actionSquare_4 = QtWidgets.QAction(MainWindow)
        self.actionSquare_4.setObjectName("actionSquare_4")
        self.actionSquare_6 = QtWidgets.QAction(MainWindow)
        self.actionSquare_6.setObjectName("actionSquare_6")
        self.actionCross_3 = QtWidgets.QAction(MainWindow)
        self.actionCross_3.setObjectName("actionCross_3")
        self.actionSquare_9 = QtWidgets.QAction(MainWindow)
        self.actionSquare_9.setObjectName("actionSquare_9")
        self.actionSquare_10 = QtWidgets.QAction(MainWindow)
        self.actionSquare_10.setObjectName("actionSquare_10")
        self.actionAritmatika = QtWidgets.QAction(MainWindow)
        self.actionAritmatika.setObjectName("actionAritmatika")
        self.menuAritmetical_Operation.addAction(self.actionAritmatika)
        self.actionBukaFile = QtWidgets.QAction(MainWindow)
        self.actionBukaFile.setObjectName("actionBukaFile")
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionBukaFile)
        self.menuFile.addAction(self.simpanSebagai)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionKeluar)
        self.menuHistogram.addAction(self.actionInput)
        self.menuHistogram.addAction(self.actionOutput)
        self.menuHistogram.addAction(self.actionInput_Output)
        self.menuView.addAction(self.menuHistogram.menuAction())
        self.menuRGB.addAction(self.actionKuning)
        self.menuRGB.addAction(self.actionOrange)
        self.menuRGB.addAction(self.actionCyan)
        self.menuRGB.addAction(self.actionPurple)
        self.menuRGB.addAction(self.actionGrey)
        self.menuRGB.addAction(self.actionCoklat)
        self.menuRGB.addAction(self.actionMerah)
        self.menuRGB_to_Grayscale.addAction(self.actionAverage)
        self.menuRGB_to_Grayscale.addAction(self.actionLightness)
        self.menuRGB_to_Grayscale.addAction(self.actionLuminance)
        self.menuBit_Depth.addAction(self.action1_Bit)
        self.menuBit_Depth.addAction(self.action2_Bit)
        self.menuBit_Depth.addAction(self.action3_Bit)
        self.menuBit_Depth.addAction(self.action4_Bit)
        self.menuBit_Depth.addAction(self.action5_Bit)
        self.menuBit_Depth.addAction(self.action6_Bit)
        self.menuBit_Depth.addAction(self.action7_Bit)
        self.menuColor.addAction(self.menuRGB.menuAction())
        self.menuColor.addAction(self.menuRGB_to_Grayscale.menuAction())
        self.menuColor.addAction(self.actionBrightness)
        self.menuColor.addAction(self.actionContrast)

        # self.menuColor.addAction(self.menuBrightness.menuAction())
        self.menuColor.addAction(self.actionInvers)
        self.menuColor.addAction(self.actionthreshold)
        self.menuColor.addAction(self.actionLog_Brightness)
        self.menuColor.addAction(self.menuBit_Depth.menuAction())
        self.menuColor.addAction(self.actionGamma_Correction)
        self.menuImage_Processing.addAction(self.actionHistogram_Equalization)
        self.menuImage_Processing.addAction(self.actionFuzzy_HE_RGB)
        self.menuImage_Processing.addAction(self.actionFuzzy_Grayscale)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_Robert)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_Sobel)
        self.menuEdge_Detection.addAction(self.actionEdge_Detection_Prewit)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_3x3)
        self.menuGaussian_Blur.addAction(self.actionGaussian_Blur_5_5)
        self.menuIdentity.addAction(self.menuEdge_Detection.menuAction())
        self.menuIdentity.addAction(self.actionSharpen)
        self.menuIdentity.addAction(self.menuGaussian_Blur.menuAction())
        self.menuIdentity.addAction(self.actionUnsharp_Masking)
        self.menuIdentity.addAction(self.actionIdentity)
        self.menuIdentity.addAction(self.actionLow_Pass_Filter)
        self.menuIdentity.addAction(self.actionHigh_Pass_Filter)
        self.menuIdentity.addAction(self.actionBandstop_Filter)
        self.menuEdge_Detection_2.addAction(self.actionPrewitt)
        self.menuErosion.addAction(self.actionSquare_3)
        self.menuErosion.addAction(self.actionSquare_5)
        self.menuErosion.addAction(self.actionCros)
        self.menuDilation.addAction(self.actionSquare_4)
        self.menuDilation.addAction(self.actionSquare_6)
        self.menuDilation.addAction(self.actionCross_3)
        self.menuOpening.addAction(self.actionSquare_9)
        self.menuClosing.addAction(self.actionSquare_10)
        self.menuMorfologi.addAction(self.menuErosion.menuAction())
        self.menuMorfologi.addAction(self.menuDilation.menuAction())
        self.menuMorfologi.addAction(self.menuOpening.menuAction())
        self.menuMorfologi.addAction(self.menuClosing.menuAction())

        # ? geometri
        self.menuGeometri = QtWidgets.QMenu(self.menubar)
        self.menuGeometri.setObjectName("menuGeometri")
        self.actionScalingUniform = QtWidgets.QAction(MainWindow)
        self.actionScalingUniform.setObjectName("actionScalingUniform")
        self.actionScalingNonUniform = QtWidgets.QAction(MainWindow)
        self.actionScalingNonUniform.setObjectName("actionScalingNonUniform")
        self.actionCropping = QtWidgets.QAction(MainWindow)
        self.actionCropping.setObjectName("actionCropping")
        self.menuFlipping = QtWidgets.QMenu(self.menuGeometri)
        self.menuFlipping.setObjectName("menuFlipping")
        self.menuGeometri.addAction(self.menuFlipping.menuAction())
        self.menuFlipping.addAction(self.actionFlippingHorizontal)
        self.menuFlipping.addAction(self.actionFlippingVertical)
        self.actionTranslasi = QtWidgets.QAction(MainWindow)
        self.actionTranslasi.setObjectName("actionTranslasi")
        self.actionRotasi = QtWidgets.QAction(MainWindow)
        self.actionRotasi.setObjectName("actionRotasi")

        self.menuGeometri.addAction(self.actionScalingUniform)
        self.menuGeometri.addAction(self.actionScalingNonUniform)
        self.menuGeometri.addAction(self.actionCropping)
        self.menuGeometri.addAction(self.actionTranslasi)
        self.menuGeometri.addAction(self.actionRotasi)

        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuView.menuAction())
        self.menubar.addAction(self.menuColor.menuAction())
        self.menubar.addAction(self.menuTentang.menuAction())
        self.menubar.addAction(self.menuImage_Processing.menuAction())
        self.menubar.addAction(self.menuAritmetical_Operation.menuAction())
        self.menubar.addAction(self.menuIdentity.menuAction())
        self.menubar.addAction(self.menuEdge_Detection_2.menuAction())
        self.menubar.addAction(self.menuMorfologi.menuAction())
        self.menubar.addAction(self.menuGeometri.menuAction())

        # todo start tambahan saya
        self.actionBukaFile.triggered.connect(self.buka_file)
        self.simpanSebagai.triggered.connect(self.simpan_sebagai)
        self.actionKeluar.triggered.connect(self.keluar_aplikasi)
        self.actionAverage.triggered.connect(self.grayscale_average)
        self.actionLightness.triggered.connect(self.greyscale_lightness)
        self.actionLuminance.triggered.connect(self.greyscale_luminance)
        self.actionAritmatika.triggered.connect(self.open_aritmatical)
        self.actionScalingUniform.triggered.connect(self.scalingUniform)
        self.actionScalingNonUniform.triggered.connect(self.scalingNonUniform)
        self.actionCropping.triggered.connect(self.cropping)
        self.actionFlippingHorizontal.triggered.connect(self.flipHorizontal)
        self.actionFlippingVertical.triggered.connect(self.flipVertical)
        self.actionTranslasi.triggered.connect(self.translasi)
        self.actionRotasi.triggered.connect(self.rotasi)
        self.actionInput.triggered.connect(self.histogram_input)
        self.actionOutput.triggered.connect(self.histogram_output)
        self.actionInput_Output.triggered.connect(self.histogram_input_output)
        self.actionBrightness.triggered.connect(self.brightnes)
        self.actionContrast.triggered.connect(self.contrast)
        self.actionInvers.triggered.connect(self.invers)
        self.actionthreshold.triggered.connect(self.threshold)
        self.actionFuzzy_HE_RGB.triggered.connect(self.fhe_rgb)
        self.actionFuzzy_Grayscale.triggered.connect(self.fhe_grayscale)
        self.actionLow_Pass_Filter.triggered.connect(self.low_pass_filter)
        self.actionHigh_Pass_Filter.triggered.connect(self.high_pass_filter)
        self.actionGaussian_Blur_3x3.triggered.connect(self.gaussian3x3)
        self.actionGaussian_Blur_5_5.triggered.connect(self.gaussian5x5)
        self.actionIdentity.triggered.connect(self.identity)
        self.actionSharpen.triggered.connect(self.sharpen)
        self.actionUnsharp_Masking.triggered.connect(self.unsharp_masking)

        self.actionEdge_Detection_Robert.triggered.connect(
            self.edge_detection_robert)
        self.actionEdge_Detection_Prewit.triggered.connect(
            self.edge_detection_prewit)
        self.actionEdge_Detection_Sobel.triggered.connect(
            self.edge_detection_sobel)
        self.actionHistogram_Equalization.triggered.connect(
            self.histogram_equalization)
        self.action1_Bit.triggered.connect(
            partial(self.bit_depth, 1))
        self.action2_Bit.triggered.connect(
            partial(self.bit_depth, 2))
        self.action3_Bit.triggered.connect(
            partial(self.bit_depth, 3))
        self.action4_Bit.triggered.connect(
            partial(self.bit_depth, 4))
        self.action5_Bit.triggered.connect(
            partial(self.bit_depth, 5))
        self.action6_Bit.triggered.connect(
            partial(self.bit_depth, 6))
        self.action7_Bit.triggered.connect(
            partial(self.bit_depth, 7))
        # todo end tambahan saya

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuView.setTitle(_translate("MainWindow", "View"))
        self.menuHistogram.setTitle(_translate("MainWindow", "Histogram"))
        self.menuColor.setTitle(_translate("MainWindow", "Color"))
        self.menuRGB.setTitle(_translate("MainWindow", "RGB"))
        self.menuRGB_to_Grayscale.setTitle(
            _translate("MainWindow", "RGB to Grayscale"))
        self.menuFlipping.setTitle(
            _translate("MainWindow", "Flipping"))
        self.actionBrightness.setText(
            _translate("MainWindow", "Brightness"))
        self.menuBit_Depth.setTitle(_translate("MainWindow", "Bit Depth"))
        self.menuTentang.setTitle(_translate("MainWindow", "Tentang"))
        self.menuImage_Processing.setTitle(
            _translate("MainWindow", "Image Processing"))
        self.menuAritmetical_Operation.setTitle(
            _translate("MainWindow", "Aritmetical Operasion"))
        self.menuIdentity.setTitle(_translate("MainWindow", "Konvolusi"))
        self.menuEdge_Detection.setTitle(
            _translate("MainWindow", "Edge Detection"))
        self.menuGaussian_Blur.setTitle(
            _translate("MainWindow", "Gaussian Blur"))
        self.menuEdge_Detection_2.setTitle(
            _translate("MainWindow", "Edge Detection"))
        self.menuMorfologi.setTitle(_translate("MainWindow", "Morfologi"))
        self.menuGeometri.setTitle(_translate("MainWindow", "Geometri"))
        self.menuErosion.setTitle(_translate("MainWindow", "Erosion"))
        self.menuDilation.setTitle(_translate("MainWindow", "Dilation"))
        self.menuOpening.setTitle(_translate("MainWindow", "Opening"))
        self.menuClosing.setTitle(_translate("MainWindow", "Closing"))
        self.simpanSebagai.setText(_translate("MainWindow", "Simpan sebagai"))
        self.actionKeluar.setText(_translate("MainWindow", "Keluar"))
        self.actionContrast.setText(
            _translate("MainWindow", "Contrast"))
        self.actionInvers.setText(_translate("MainWindow", "Invers"))
        self.actionthreshold.setText(_translate("MainWindow", "Threshold"))
        self.actionLog_Brightness.setText(
            _translate("MainWindow", "Log Brightness"))
        self.actionGamma_Correction.setText(
            _translate("MainWindow", "Gamma Correction"))
        self.actionHistogram_Equalization.setText(
            _translate("MainWindow", "Histogram Equalization"))
        self.actionFuzzy_HE_RGB.setText(
            _translate("MainWindow", "Fuzzy HE RGB"))
        self.actionFuzzy_Grayscale.setText(
            _translate("MainWindow", "Fuzzy Grayscale"))
        self.actionSharpen.setText(_translate("MainWindow", "Sharpen"))
        self.actionUnsharp_Masking.setText(
            _translate("MainWindow", "Unsharp Masking"))
        self.actionIdentity.setText(
            _translate("MainWindow", "Identity"))
        self.actionLow_Pass_Filter.setText(
            _translate("MainWindow", "Low Pass Filter"))
        self.actionHigh_Pass_Filter.setText(
            _translate("MainWindow", "High Pass Filter"))
        self.actionBandstop_Filter.setText(
            _translate("MainWindow", "Bandstop Filter"))

        self.actionRotasi.setText(
            _translate("MainWindow", "Rotasi"))
        self.actionTranslasi.setText(
            _translate("MainWindow", "Translasi"))
        self.actionScalingNonUniform.setText(
            _translate("MainWindow", "Scaling Non Uniform"))
        self.actionCropping.setText(_translate("MainWindow", "Cropping"))
        self.actionFlippingHorizontal.setText(
            _translate("MainWindow", "Flipping Horizontal"))
        self.actionFlippingVertical.setText(
            _translate("MainWindow", "Flipping Vertical"))
        self.actionKuning.setText(_translate("MainWindow", "Kuning"))
        self.actionOrange.setText(_translate("MainWindow", "Orange"))
        self.actionCyan.setText(_translate("MainWindow", "Cyan"))
        self.actionPurple.setText(_translate("MainWindow", "Purple"))
        self.actionGrey.setText(_translate("MainWindow", "Grey"))
        self.actionCoklat.setText(_translate("MainWindow", "Coklat"))
        self.actionMerah.setText(_translate("MainWindow", "Merah"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        self.actionLightness.setText(_translate("MainWindow", "Lightness"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.action1_Bit.setText(_translate("MainWindow", "1 Bit"))
        self.action2_Bit.setText(_translate("MainWindow", "2 Bit"))
        self.action3_Bit.setText(_translate("MainWindow", "3 Bit"))
        self.action4_Bit.setText(_translate("MainWindow", "4 Bit"))
        self.action5_Bit.setText(_translate("MainWindow", "5 Bit"))
        self.action6_Bit.setText(_translate("MainWindow", "6 Bit"))
        self.action7_Bit.setText(_translate("MainWindow", "7 Bit"))
        self.actionInput.setText(_translate("MainWindow", "Input"))
        self.actionOutput.setText(_translate("MainWindow", "Output"))
        self.actionInput_Output.setText(
            _translate("MainWindow", "Input Output"))
        self.actionEdge_Detection_Robert.setText(
            _translate("MainWindow", "Edge Detection Robert"))
        self.actionEdge_Detection_Sobel.setText(
            _translate("MainWindow", "Edge Detection Sobel"))
        self.actionEdge_Detection_Prewit.setText(
            _translate("MainWindow", "Edge Detection Prewit"))
        self.actionGaussian_Blur_3x3.setText(
            _translate("MainWindow", "Gaussian Blur 3x3"))
        self.actionGaussian_Blur_5_5.setText(
            _translate("MainWindow", "Gaussian Blur 5x5"))
        self.actionPrewitt.setText(_translate("MainWindow", "Prewitt"))
        self.actionSquare_3.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_5.setText(_translate("MainWindow", "Square 5"))
        self.actionCros.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_4.setText(_translate("MainWindow", "Square 3"))
        self.actionSquare_6.setText(_translate("MainWindow", "Square 5"))
        self.actionCross_3.setText(_translate("MainWindow", "Cross 3"))
        self.actionSquare_9.setText(_translate("MainWindow", "Square 9"))
        self.actionSquare_10.setText(_translate("MainWindow", "Square 9"))
        self.actionBukaFile.setText(_translate("MainWindow", "Buka file"))
        self.actionAritmatika.setText(_translate("MainWindow", "Aritmatika"))
        self.actionScalingUniform.setText(
            _translate('MainWindow', 'Scaling Uniform'))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
