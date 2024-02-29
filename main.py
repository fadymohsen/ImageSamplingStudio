import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget
from PyQt5 import uic






class MyTabWidget(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        uic.loadUi(ui_file, self)

    def keyPressEvent(self, event):
        if event.key() == 16777216:  # Integer value for Qt.Key_Escape
            if self.isFullScreen():
                self.showNormal()  # Show in normal mode
            else:
                self.showFullScreen()  # Show in full screen
        else:
            super().keyPressEvent(event)






def main():
    app = QApplication(sys.argv)
    window = MyTabWidget("MainWindow.ui")
    window.showFullScreen()                         # Show in full screen

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
