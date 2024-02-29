from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap
import sys

class ImageWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Load an image file into a QPixmap
        pixmap = QPixmap("path/to/your/image.jpg")

        # Create a QLabel widget and set the pixmap as the image
        label = QLabel(self)
        label.setPixmap(pixmap)
        label.setScaledContents(True)

        # Set the QLabel as the central widget of the window
        self.setCentralWidget(label)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageWindow()
    window.show()
    sys.exit(app.exec_())
