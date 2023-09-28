
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import sys  # Import modul sys
            
class kedua(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(993, 664)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 120, 471, 291))
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setLineWidth(2)
        self.label.setText("")
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(500, 120, 471, 291))
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setLineWidth(2)
        self.label_2.setText("")
        self.label_2.setScaledContents(True)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 993, 26))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuImage = QtWidgets.QMenu(self.menubar)
        self.menuImage.setObjectName("menuImage")
        self.menuRGB_to_GreyScale = QtWidgets.QMenu(self.menuImage)
        self.menuRGB_to_GreyScale.setObjectName("menuRGB_to_GreyScale")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.Lightness = QtWidgets.QAction(MainWindow)
        self.Lightness.setObjectName("Lightness")
        self.Lightness.triggered.connect(self.showLightness)
        self.actionOpen = QtWidgets.QAction(MainWindow)
        self.actionOpen.setObjectName("actionOpen")
        # fungsi open file , Luminance
        self.actionOpen.triggered.connect(self.openFile)
        self.actionSave_As = QtWidgets.QAction(MainWindow)
        self.actionSave_As.setObjectName("actionSave_As")
        # fungsi simpan gambar
        self.actionSave_As.triggered.connect(self.saveImage)
        # ===============================================
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        # fungsi keluar daari aplikasi
        self.actionExit.triggered.connect(self.exitApplication) 
        # ====================================================
        self.actionLuminance = QtWidgets.QAction(MainWindow)
        self.actionLuminance.setObjectName("actionLuminance")
        self.actionLuminance.triggered.connect(self.showLuminance)
        self.actionAverage = QtWidgets.QAction(MainWindow)
        self.actionAverage.setObjectName("actionAverage")
        self.actionAverage.triggered.connect(self.showAverage)
        self.menuFile.addSeparator()
        self.menuFile.addAction(self.actionOpen)
        self.menuFile.addAction(self.actionSave_As)
        self.menuFile.addAction(self.actionExit)
        self.menuRGB_to_GreyScale.addAction(self.Lightness)
        self.menuRGB_to_GreyScale.addAction(self.actionLuminance)
        self.menuRGB_to_GreyScale.addAction(self.actionAverage)
        self.menuImage.addSeparator()
        self.menuImage.addAction(self.menuRGB_to_GreyScale.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())
        self.menubar.addAction(self.menuImage.menuAction())
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        
    # digunakan untuk membuat UI
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Ps Beta V-1"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuImage.setTitle(_translate("MainWindow", "Image"))
        self.menuRGB_to_GreyScale.setTitle(_translate("MainWindow", "RGB to GreyScale"))
        self.Lightness.setText(_translate("MainWindow", "Lightness"))
        self.actionOpen.setText(_translate("MainWindow", "Open"))
        self.actionSave_As.setText(_translate("MainWindow", "Save As"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionLuminance.setText(_translate("MainWindow", "Luminance"))
        self.actionAverage.setText(_translate("MainWindow", "Average"))
        
    # Membuat fungsi untuk membuka file pada python    
    def openFile(self):
        options = QFileDialog.Options() # Inisialisasi opsi untuk dialog pemilihan berkas
        options |= QFileDialog.ReadOnly # Menambahkan opsi mode baca saja ke dalam opsi dialog
        # menghapus gambar dari label kedua
        self.label_2.clear()
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
    
    # Membuat fungsi untuk merubah warna gambar        
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
        self.turu = grayscale_image       
    
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
        self.turu = average  
        
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
        self.turu = lightness 
                
    def saveImage(self):
        if hasattr(self, 'turu'):
            # Inisialisasi opsi untuk dialog pemilihan berkas
            options = QFileDialog.Options()
            # Menambahkan opsi mode baca saja ke dalam opsi dialog
            options |= QFileDialog.ReadOnly 
            # menampung file path dari dialog open file dan difilter hanya format png , jpg , bmp
            file_name, _ = QFileDialog.getSaveFileName(None, "Save Image File", "", "Images (*.png *.jpg *.bmp *.jpeg);;All Files (*)", options=options)
            # check apakah terdapat path file
            if file_name:
                #Simpan gambar yang telah diformat
                self.turu.save(file_name)   
                       
    def exitApplication(self):
        # untuk keluar dari aplikasi
        QtWidgets.qApp.quit()     
           
if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    uiBalap = kedua()
    uiBalap.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
