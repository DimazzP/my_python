import sys
import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton


class ROIApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ROI Selector")
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(10, 10, 640, 480)

        self.select_button = QPushButton("Select ROI", self)
        self.select_button.setGeometry(660, 10, 120, 30)
        self.select_button.clicked.connect(self.select_roi)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(660, 50, 120, 30)

        self.image = None
        self.selected_roi = None

    def select_roi(self):
        self.selected_roi = cv2.selectROI('Select Area', self.image)
        if all(self.selected_roi):
            roi = self.image[int(self.selected_roi[1]):int(self.selected_roi[1] + self.selected_roi[3]),
                             int(self.selected_roi[0]):int(self.selected_roi[0] + self.selected_roi[2])]
            cv2.imshow('ROI', roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            self.result_label.setText(f'Selected ROI: {self.selected_roi}')
        else:
            self.result_label.setText('No ROI selected')

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        self.display_image()

    def display_image(self):
        if self.image is not None:
            height, width, channel = self.image.shape
            bytes_per_line = 3 * width
            q_image = QImage(self.image.data, width, height,
                             bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ROIApp()
    window.load_image('jeruk.jpeg')
    window.show()
    sys.exit(app.exec_())
