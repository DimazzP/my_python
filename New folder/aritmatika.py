
from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

class Ui_Aritmatika(object):
    def setupUi(self, Aritmatika):
        Aritmatika.setObjectName("Aritmatika")
        Aritmatika.resize(910, 351)
        self.centralwidget = QtWidgets.QWidget(Aritmatika)
        self.centralwidget.setObjectName("centralwidget")
        self.gambar1 = QtWidgets.QLabel(self.centralwidget)
        self.gambar1.setGeometry(QtCore.QRect(14, 90, 281, 171))
        self.gambar1.setFrameShape(QtWidgets.QFrame.Box)
        self.gambar1.setLineWidth(2)
        self.gambar1.setText("")
        self.gambar1.setScaledContents(True)
        self.gambar1.setObjectName("gambar1")
        self.gambar2 = QtWidgets.QLabel(self.centralwidget)
        self.gambar2.setGeometry(QtCore.QRect(310, 90, 281, 171))
        self.gambar2.setFrameShape(QtWidgets.QFrame.Box)
        self.gambar2.setLineWidth(2)
        self.gambar2.setText("")
        self.gambar2.setScaledContents(True)
        self.gambar2.setObjectName("gambar2")
        self.hasil = QtWidgets.QLabel(self.centralwidget)
        self.hasil.setGeometry(QtCore.QRect(610, 90, 281, 171))
        self.hasil.setFrameShape(QtWidgets.QFrame.Box)
        self.hasil.setLineWidth(2)
        self.hasil.setText("")
        self.hasil.setScaledContents(True)
        self.hasil.setObjectName("hasil")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 70, 55, 16))
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(310, 70, 55, 16))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(610, 70, 101, 16))
        self.label_3.setObjectName("label_3")
        Aritmatika.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Aritmatika)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 910, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuAritmatika = QtWidgets.QMenu(self.menubar)
        self.menuAritmatika.setObjectName("menuAritmatika")
        Aritmatika.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Aritmatika)
        self.statusbar.setObjectName("statusbar")
        Aritmatika.setStatusBar(self.statusbar)
        self.actionOpen = QtWidgets.QAction(Aritmatika)
        self.actionOpen.setObjectName("actionOpen")
        self.actionOpen.triggered.connect(self.open_images)
        self.actionSave = QtWidgets.QAction(Aritmatika)
        self.actionSave.setObjectName("actionSave")
        self.actionSave.triggered.connect(self.saveImage)
        self.actionAND = QtWidgets.QAction(Aritmatika)
        self.actionAND.setObjectName("actionAND")
        self.actionAND.triggered.connect(self.and_process)
        self.actionOR = QtWidgets.QAction(Aritmatika)
        self.actionOR.setObjectName("actionOR")
        self.actionOR.triggered.connect(self.or_process)
        self.actionNOT = QtWidgets.QAction(Aritmatika)
        self.actionNOT.setObjectName("actionNOT")
        self.actionNOT.triggered.connect(self.not_process)
        self.actionPlus = QtWidgets.QAction(Aritmatika)
        self.actionPlus.setObjectName("actionPlus")
        self.actionPlus.triggered.connect(self.plus_process)
        self.actionMin = QtWidgets.QAction(Aritmatika)
        self.actionMin.setObjectName("actionMin")
        self.actionMin.triggered.connect(self.min_process)
        self.actionKali = QtWidgets.QAction(Aritmatika)
        self.actionKali.setObjectName("actionKali")
        self.actionKali.triggered.connect(self.kali_process)
        self.actionBagi = QtWidgets.QAction(Aritmatika)
        self.actionBagi.setObjectName("actionBagi")
        self.actionBagi.triggered.connect(self.pembagian_process)
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave)
        self.menuAritmatika.addAction(self.actionAND)
        self.menuAritmatika.addAction(self.actionOR)
        self.menuAritmatika.addAction(self.actionNOT)
        self.menuAritmatika.addAction(self.actionPlus)
        self.menuAritmatika.addAction(self.actionMin)
        self.menuAritmatika.addAction(self.actionKali)
        self.menuAritmatika.addAction(self.actionBagi)
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuAritmatika.menuAction())

        self.retranslateUi(Aritmatika)
        QtCore.QMetaObject.connectSlotsByName(Aritmatika)

    def retranslateUi(self, Aritmatika):
        _translate = QtCore.QCoreApplication.translate
        Aritmatika.setWindowTitle(_translate("Aritmatika", "MainWindow"))
        self.label.setText(_translate("Aritmatika", "Gambar 1"))
        self.label_2.setText(_translate("Aritmatika", "Gambar 2"))
        self.label_3.setText(_translate("Aritmatika", "Hasil Aritmatika"))
        self.menuFile.setTitle(_translate("Aritmatika", "File"))
        self.menuAritmatika.setTitle(_translate("Aritmatika", "Aritmatika"))
        self.actionOpen.setText(_translate("Aritmatika", "Open"))
        self.actionSave.setText(_translate("Aritmatika", "Save"))
        self.actionAND.setText(_translate("Aritmatika", "AND"))
        self.actionOR.setText(_translate("Aritmatika", "OR"))
        self.actionNOT.setText(_translate("Aritmatika", "NOT"))
        self.actionPlus.setText(_translate("Aritmatika","PLUS"))
        self.actionMin.setText(_translate("Aritmatika","MIN"))
        self.actionKali.setText(_translate("Aritmatika","KALI"))
        self.actionBagi.setText(_translate("Aritmatika","BAGI"))
        
    def open_images(self):
        options = QFileDialog.Options()
        file_paths, _ = QFileDialog.getOpenFileNames(None, "Open Images", "", "Images (*.png *.jpg *.bmp *.jpeg *.gif *.tif *.tiff);;All Files (*)", options=options)
        self.hasil.clear()
        if len(file_paths) == 2:
            self.image1 = QtGui.QImage(file_paths[0])
            self.image2 = QtGui.QImage(file_paths[1])

            pixmap1 = QtGui.QPixmap.fromImage(self.image1)
            self.gambar1.setPixmap(pixmap1)
            self.gambar1.setScaledContents(True)

            pixmap2 = QtGui.QPixmap.fromImage(self.image2)
            self.gambar2.setPixmap(pixmap2)
            self.gambar2.setScaledContents(True)
            
    def and_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))
                        
                    r_result = color1.red() & color2.red()
                    g_result = color1.green() & color2.green()
                    b_result = color1.blue() & color2.blue()

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image 
            
    def kali_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))

                    r_result = min(color1.red() * color2.red(), 255)  # Batasan nilai maksimum
                    g_result = min(color1.green() * color2.green(), 255)  # Batasan nilai maksimum
                    b_result = min(color1.blue() * color2.blue(), 255)  # Batasan nilai maksimum

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image 
            
    def pembagian_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))

                    # Hindari pembagian oleh nol atau nilai yang sangat kecil
                    divisorRed = max(color2.red(), 1)
                    divisorgreen = max(color2.green(),1)
                    divisorBlue = max(color2.blue(),1)
                    
                    r_result = max(round(color1.red() / divisorRed), 0)  # Batasan nilai maksimum
                    g_result = max(round(color1.green() / divisorgreen), 0)  # Batasan nilai maksimum
                    b_result = max(round(color1.blue() / divisorBlue), 0)  # Batasan nilai maksimum

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image 
                           
    def min_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))

                    r_result = max(color1.red() - color2.red(), 0)  # Batasan nilai minimum
                    g_result = max(color1.green() - color2.green(), 0)  # Batasan nilai minimum
                    b_result = max(color1.blue() - color2.blue(), 0)  # Batasan nilai minimum

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image 
                    
    def plus_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))

                    r_result = min(color1.red() + color2.red(),255)
                    g_result = min(color1.green() + color2.green(),255)
                    b_result = min(color1.blue() + color2.blue(),255)

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image 
                    
    def or_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))

                    r_result = color1.red() | color2.red()
                    g_result = color1.green() | color2.green()
                    b_result = color1.blue() | color2.blue()

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image 
            
    def not_process(self):
        if self.image1 is not None and self.image2 is not None:
            width = min(self.image1.width(), self.image2.width())
            height = min(self.image1.height(), self.image2.height())
            result_image = QImage(width, height, QImage.Format_RGB888)

            for x in range(width):
                for y in range(height):
                    color1 = QtGui.QColor(self.image1.pixelColor(x, y))
                    color2 = QtGui.QColor(self.image2.pixelColor(x, y))

                    r_result = 255 - color2.red()
                    g_result = 255 - color2.green()
                    b_result = 255 - color2.blue()

                    result_pixel = QtGui.qRgb(r_result, g_result, b_result)
                    result_image.setPixel(x, y, result_pixel)

            result_pixmap = QPixmap.fromImage(result_image)
            self.hasil.setPixmap(result_pixmap)
            self.hasil.setScaledContents(True)
            self.hasilGambar = result_image  
               
    def saveImage(self):
        if hasattr(self, 'hasilGambar'):
            # Inisialisasi opsi untuk dialog pemilihan berkas
            options = QFileDialog.Options()
            # Menambahkan opsi mode baca saja ke dalam opsi dialog
            options |= QFileDialog.ReadOnly 
            # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
            file_name, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
            # check apakah terdapat path file
            if file_name:
                #Simpan gambar yang telah diformat
                self.hasilGambar.save(file_name)  
                                 
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Aritmatika = QtWidgets.QMainWindow()
    ui = Ui_Aritmatika()
    ui.setupUi(Aritmatika)
    Aritmatika.show()
    sys.exit(app.exec_())
