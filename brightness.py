from PyQt5.QtWidgets import QDialog, QVBoxLayout, QPushButton, QLabel, QSlider
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage, QColor


class BrightnessDialog(QDialog):
    # Buat sinyal untuk mengirimkan nilai slider ke kelas utama
    sliderValueChanged = pyqtSignal(int)

    def __init__(self, parent=None, main_window=None):
        super().__init__(parent)

        self.main_window = main_window

        layout = QVBoxLayout(self)

        self.slider = QSlider(self)
        self.slider.setOrientation(Qt.Horizontal)
        self.slider.setMinimum(-255)
        self.slider.setMaximum(255)
        self.slider.setValue(0)
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setText("0")

        # Tambahkan tombol "OK"
        self.ok_button = QPushButton("OK", self)
        self.ok_button.clicked.connect(self.onOkButtonClicked)

        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        layout.addWidget(self.ok_button)

        self.slider.valueChanged.connect(self.onSliderChange)

    def onOkButtonClicked(self):
        if self.main_window:
            pixmap = self.main_window.label_gambar_asal.pixmap()
            image = pixmap.toImage()
            width = image.width()
            height = image.height()

            for y in range(height):
                for x in range(width):
                    pixel_color = image.pixelColor(x, y)
                    r, g, b = pixel_color.red(), pixel_color.green(), pixel_color.blue()
                    r = min(max(r + self.slider.value(), 0), 255)
                    g = min(max(g + self.slider.value(), 0), 255)
                    b = min(max(b + self.slider.value(), 0), 255)
                    new_color = QColor(r, g, b)
                    image.setPixelColor(x, y, new_color)

            pixmap_grayscale = QPixmap.fromImage(image)
            self.main_window.label_gambar_tujuan.setPixmap(pixmap_grayscale)
            self.main_window.label_gambar_tujuan.setScaledContents(True)

    def getValue(self):
        print('custom')
        print(self.slider.value())
        return self.slider.value()

    def onSliderChange(self, value):
        self.label.setText(str(value))

    # def onOkButtonClicked(self):
    #     # Ketika tombol "OK" ditekan, kirim nilai slider melalui sinyal
    #     self.sliderValueChanged.emit(self.slider.value())
