import sys
from PyQt5.QtWidgets import QApplication, QTabWidget, QFileDialog
from PyQt5 import uic
import cv2
import pyqtgraph as pg
import numpy as np

# Features
from features.NoiseFilter import noiseAdditionFiltration
from features.EdgeDetection import EdgeDetector
from features.Thresholding import Thresholding 
from features.curves import Curves
from features.normalizeAndEqualize import ImageProcessor
from features.frequency_domain_filters import FrequencyDomainFilters
from features.RGBHistogram import RGBHistograms





# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------





class MyTabWidget(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        uic.loadUi(ui_file, self)
        self.image_edges = pg.GraphicsLayoutWidget()
        self.selected_image_path = None
        self.pushButton_browseImage.clicked.connect(self.browse_image)

        # Import Features Classes
        self.noiseAddFilterAdd = noiseAdditionFiltration(self)
        self.addDetectionAdd = EdgeDetector(self)
        self.addThresholdingAdd = Thresholding(self)
        self.addCurvesAdd = Curves(self)
        self.addEqualizeNormalize = ImageProcessor(self)
        self.frequency_domain_filters = FrequencyDomainFilters(self) 
        self.RGBHistograms = RGBHistograms(self) 
        

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def browse_image(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
        if file_name:
            self.selected_image_path = file_name
            self.display_image_on_graphics_layout(file_name)
            # Apply in Noise/Filtering
            self.noiseAddFilterAdd.applyNoise()
            # Apply in EdgeDetection
            self.addDetectionAdd.detectEdges()
            # Apply in curves
            self.addCurvesAdd.drawCurves()
            # Apply in Normalization & Equalization
            self.addEqualizeNormalize.imageProcessing()
            # Apply in frequency domain filters
            self.frequency_domain_filters.apply_frequency_domain_filters()
            self.RGBHistograms.handleButton()
            self.addThresholdingAdd.handleButton1()


# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def display_image_on_graphics_layout(self, image_path):
        image_data = cv2.imread(image_path)
        image_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
        image_data = np.rot90(image_data, -1)
        # Clear the previous image if any
        self.graphicsLayoutWidget_displayImagesMain.clear()
        # Create a PlotItem or ViewBox
        view_box = self.graphicsLayoutWidget_displayImagesMain.addViewBox()
        # Create an ImageItem and add it to the ViewBox
        image_item = pg.ImageItem(image_data)
        view_box.addItem(image_item)
        # Optional: Adjust the view to fit the image
        view_box.autoRange()

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def keyPressEvent(self, event):
        if event.key() == 16777216:         # Integer value for Qt.Key_Escape
            if self.isFullScreen():
                self.showNormal()           # Show in normal mode
            else:
                self.showFullScreen()       # Show in full screen
        else:
            super().keyPressEvent(event)
    
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------


def main():
    app = QApplication(sys.argv)
    window = MyTabWidget("MainWindow.ui")
    window.showFullScreen()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()