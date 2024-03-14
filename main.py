import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget, QFileDialog
from PyQt5 import uic
import cv2
import pyqtgraph as pg
from frequency_domain_filters import apply_convolution, ideal_filter, butterworth_filter, gaussian_filter, hyprid_images
import matplotlib.pyplot as plt
import numpy as np
import curves
import EdgeDetection
from Noise import noiseAddition



class MyTabWidget(QTabWidget):
    def __init__(self, ui_file):
        super().__init__()
        uic.loadUi(ui_file, self)
        self.edgeDetectionDirection = "Horizontal"
        self.currentTypeIndex = 0
        self.handleObjects()
        self.image_data = None
        self.vb = None
        self.filter_types = {"Ideal":ideal_filter,
        "Butterworth":butterworth_filter,
        "Gaussian":gaussian_filter}
        self.image_mixed = False
        self.img_data_low_pass = None
        self.img_data_high_pass  = None
        self.counter = 0
        self.noiseAdd = noiseAddition(self)


    def keyPressEvent(self, event):
        if event.key() == 16777216:  # Integer value for Qt.Key_Escape
            if self.isFullScreen():
                self.showNormal()  # Show in normal mode
            else:
                self.showFullScreen()  # Show in full screen
        else:
            super().keyPressEvent(event)
    
    
    def handleObjects(self):
        self.btn_chooseImageCurves_2.clicked.connect(self.open_image)
        self.btn_chooseImageNoise_2.clicked.connect(lambda: self.noiseAdd.Browse())
        self.slider_adjustFrequency.valueChanged.connect(self.updateFrequencyValue)
        self.slider_adjustTValue.valueChanged.connect(self.updateTValue)
        self.radioButton_highPass.clicked.connect(
            lambda: self.freq_domain_filters(self.radioButton_lowPass.isChecked()))
        self.radioButton_lowPass.clicked.connect(
            lambda: self.freq_domain_filters(self.radioButton_lowPass.isChecked()))
        self.comboBox_filterType.currentTextChanged.connect(
            lambda: self.freq_domain_filters(self.radioButton_lowPass.isChecked()
                                             , self.comboBox_filterType.currentText()))
        self.btn_addFirstImage.clicked.connect(self.toggle_data)
        self.btn_addSecondImage.clicked.connect(self.toggle_data)
        self.btn_addFirstImage.clicked.connect(
            lambda: self.freq_domain_filters(True,
                                             self.comboBox_filterType.currentText()))
        self.btn_addSecondImage.clicked.connect(lambda:self.freq_domain_filters(False,
                                                                                self.comboBox_filterType.currentText()))

        self.btn_applyHybrid.clicked.connect(lambda:self.mix_images(self.img_data_low_pass,self.img_data_high_pass))
        self.comboBox_edgeMaskType.currentIndexChanged.connect(self.on_edgeMaskType_change)
        self.comboBox_edgeMaskDirection.currentIndexChanged.connect(self.on_edgeMaskDirection_change)

    def updateFrequencyValue(self, value):
        self.label_frequencyValue.setText('Cut-off Frequency: {} Hz'.format(value))
        self.freq_domain_filters(self.radioButton_lowPass.isChecked()
                                 , self.comboBox_filterType.currentText())

    def updateTValue(self, value):
        self.label_valueOfT.setText('T-Value: {}'.format(value))

    def toggle_data(self):
        self.image_mixed = True
        self.open_image()

    def open_image(self):
        options = QFileDialog.Options()
        fileName, _ = QFileDialog.getOpenFileName(self, "Select Image", "",
                                                  "Image Files (*.png *.jpg *.jpeg *.bmp *.gif)",
                                                  options=options)
        imageEdgeDetection = cv2.imread(fileName)
        imageEdgeDetection = cv2.rotate(imageEdgeDetection, cv2.ROTATE_90_CLOCKWISE)
        self.image_data = cv2.imread(fileName, 0)
        self.read_image(self.image_beforeEdgeDetection, self.image_data)
        self.grayScaleImage = cv2.cvtColor(imageEdgeDetection, cv2.COLOR_BGR2GRAY)
        histogramCurve, distributionCurve = curves.drawCurves(self.image_data)

        view = self.image_histogramCurve.addViewBox()
        view.addItem(histogramCurve)
        distributionView = self.image_distributionCurve.addViewBox()
        distributionView.addItem(distributionCurve )
        self.checkMaskType()

        if not self.image_mixed:
            self.read_image(self.image_originalFrequencies, self.image_data)

    def on_edgeMaskType_change(self, index):
        self.currentTypeIndex = index
        self.checkMaskType()

    def checkMaskType(self):
        if self.currentTypeIndex == 0:
            img = EdgeDetection.sobelEdgeDetection(self.grayScaleImage, self.edgeDetectionDirection)
        elif self.currentTypeIndex == 1:
            img = EdgeDetection.prewittEdgeDetection(self.grayScaleImage, self.edgeDetectionDirection)
        elif self.currentTypeIndex == 2:
            img = EdgeDetection.robertEdgeDetection(self.grayScaleImage, self.edgeDetectionDirection)
        else:
            img = EdgeDetection.cannyEdgeDetection(self.grayScaleImage)
        
        self.showFinalImage(img)


    def on_edgeMaskDirection_change(self, index):
        if index == 0:
            self.edgeDetectionDirection = "Horizontal"
            self.checkMaskType()
        else:
            self.edgeDetectionDirection = "Vertical"
            self.checkMaskType()

    def clear_view_box(self,graph_widget):
        if graph_widget is not None:
            graph_widget.clear()

    def read_image(self,graph_name, img_data):
        if self.vb is not None:
            self.clear_view_box(graph_name)
        img_data = cv2.rotate(img_data, cv2.ROTATE_90_CLOCKWISE)
        self.vb = graph_name.addViewBox()

        img = pg.ImageItem(img_data)
        self.vb.addItem(img)
        # self.vb.setAspectLocked(True)  # Lock aspect ratio to prevent scaling
        # self.vb.autoRangeEnabled = False  # Disable auto-scaling


    def freq_domain_filters(self,low_pass_applied,type_filter="Ideal"):
        cut_off_freq = int(self.slider_adjustFrequency.value())
        img_filtered_data = self.filter_types[type_filter](self.image_data, cut_off_freq, low_pass_applied)

        if self.image_mixed:
            self.counter += 1
            if low_pass_applied:
                self.img_data_low_pass = img_filtered_data
                self.read_image(self.image_lowPass, np.abs(img_filtered_data))
            else:
                self.img_data_high_pass = img_filtered_data
                self.read_image(self.image_highPass, np.abs(img_filtered_data))
        else:
            self.read_image(self.image_filteredFrequencies, np.abs(img_filtered_data))

    def mix_images(self,img_data1,img_data2):
        result = hyprid_images(img_data1,img_data2)
        self.read_image(self.image_result, result)

    def showFinalImage(self, edgeDetectedImage):
        self.image_afterEdgeDetection.clear()
        edgeDetectedImage = pg.ImageItem(edgeDetectedImage)
        view = self.image_afterEdgeDetection.addViewBox()
        view.addItem(edgeDetectedImage)



def main():
    app = QApplication(sys.argv)
    window = MyTabWidget("MainWindow.ui")
    window.showFullScreen()                         # Show in full screen

    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()