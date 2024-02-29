import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5 import uic

class MyTabWidget(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        uic.loadUi(ui_file, self)

def main():
    app = QApplication(sys.argv)
    window = MyTabWidget("MainWindow.ui")
    window.showFullScreen()                         # Show in full screen

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
