import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap, QImage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class HistogramDialog(QDialog):
    def __init__(self, image_path):
        super().__init__()
        
        self.imageObj = cv2.imread(image_path)
        self.blue_color = cv2.calcHist([self.imageObj], [0], None, [256], [0, 256])
        self.red_color = cv2.calcHist([self.imageObj], [1], None, [256], [0, 256])
        self.green_color = cv2.calcHist([self.imageObj], [2], None, [256], [0, 256])

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        label_blue = QLabel("Histogram of Blue")
        layout.addWidget(label_blue)
        self.blue_plot = self.create_histogram_plot(self.blue_color, 'blue')
        layout.addWidget(self.blue_plot)

        label_green = QLabel("Histogram of Green")
        layout.addWidget(label_green)
        self.green_plot = self.create_histogram_plot(self.green_color, 'green')
        layout.addWidget(self.green_plot)

        label_red = QLabel("Histogram of Red")
        layout.addWidget(label_red)
        self.red_plot = self.create_histogram_plot(self.red_color, 'red')
        layout.addWidget(self.red_plot)

        button_close = QPushButton("Close")
        button_close.clicked.connect(self.close)
        layout.addWidget(button_close)

        self.setLayout(layout)
        self.setWindowTitle("Histogram Viewer")

    def create_histogram_plot(self, histogram, color):
        fig = Figure()
        ax = fig.add_subplot(111)
        ax.hist(np.arange(256), bins=256, weights=histogram, color=color)
        ax.set_xlim([0, 255])
        ax.set_ylim([0, max(histogram)])
        ax.set_title(f"Histogram of {color.capitalize()}")
        canvas = FigureCanvas(fig)
        return canvas

if __name__ == '__main__':
    app = QApplication(sys.argv)

