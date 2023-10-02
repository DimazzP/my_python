import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from rembg import remove


class RemoveBackgroundApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Remove Background App')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        # self.image_label.setAlignment(Qt.AlignCenter)

        self.load_button = QPushButton('Load Image', self)
        self.load_button.clicked.connect(self.load_image)

        self.remove_bg_button = QPushButton('Remove Background', self)
        self.remove_bg_button.clicked.connect(self.remove_background)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        layout = QVBoxLayout()
        layout.addWidget(self.load_button)
        layout.addWidget(self.remove_bg_button)
        layout.addWidget(self.image_label)
        self.central_widget.setLayout(layout)

        self.image = None

    def load_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, 'Open Image File', '', 'Images (*.png *.jpg *.jpeg *.bmp *.gif);;All Files (*)', options=options)

        if file_path:
            self.image = file_path
            self.display_image(self.image)

    def display_image(self, image_path):
        if image_path:
            q_image = QImage(image_path)
            self.image_label.setPixmap(QPixmap.fromImage(q_image))

    def remove_background(self):
        if self.image:
            output_path = 'output.png'  # Ubah sesuai kebutuhan Anda
            with open(self.image, "rb") as input_file:
                with open(output_path, "wb") as output_file:
                    output_file.write(remove(input_file.read()))

            self.display_image(output_path)


def main():
    app = QApplication(sys.argv)
    window = RemoveBackgroundApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
