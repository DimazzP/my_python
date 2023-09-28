import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLineEdit


class InputWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.layout = QVBoxLayout()
        self.input_field = QLineEdit(self)
        self.submit_button = QPushButton("Submit", self)
        self.layout.addWidget(self.input_field)
        self.layout.addWidget(self.submit_button)
        self.setLayout(self.layout)
        self.submit_button.clicked.connect(self.submit_text)

    def submit_text(self):
        text = self.input_field.text()
        if text:
            self.parent().set_main_text(text)
            self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.text_label = QLineEdit(self)
        self.input_button = QPushButton("Open Input", self)
        self.layout.addWidget(self.text_label)
        self.layout.addWidget(self.input_button)
        self.central_widget.setLayout(self.layout)
        self.input_button.clicked.connect(self.open_input_window)

    def open_input_window(self):
        self.input_window = InputWindow(self)
        self.input_window.show()

    def set_main_text(self, text):
        self.text_label.setText(text)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
