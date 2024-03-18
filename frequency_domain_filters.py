import cv2
from PyQt5.QtWidgets import QFileDialog
import numpy as np
import pyqtgraph as pg

 # define filters 0 -> ideal 1-> butterworth , 2-> gaussian
 #
 # type_filters = {0:
 # ,1:,2:}
class FrequencyDomainFilters:
    def __init__(self,main_tab_widget):
        self.mainWindow = main_tab_widget
        self.image_data = None
        self.vb = None
        self.filter_types = {"Ideal":self.ideal_filter,
        "Butterworth":self.butterworth_filter,
        "Gaussian":self.gaussian_filter}
        self.image_mixed = False
        self.img_data_low_pass = None
        self.img_data_high_pass  = None
        self.counter = 0

        
    def apply_convolution(self,img, kernel):
            img = self.fourier_transform(img)
            img_after_kernal = img * kernel
            img_after_kernal = np.fft.ifft2(np.fft.ifftshift(img_after_kernal))
            img_after_kernal = np.clip(img_after_kernal, 0,255)
            return img_after_kernal.astype(np.uint8)
    

    def apply_frequency_domain_filters(self):
        self.open_image()

        self.mainWindow.slider_adjustFrequency.valueChanged.connect(self.updateFrequencyValue)
        # self.mainWindow.slider_adjustTValue.valueChanged.connect(self.updateTValue)
        self.mainWindow.radioButton_highPass.clicked.connect(
            lambda: self.freq_domain_filters(self.mainWindow.radioButton_lowPass.isChecked()))
        self.mainWindow.radioButton_lowPass.clicked.connect(
            lambda: self.freq_domain_filters(self.mainWindow.radioButton_lowPass.isChecked()))
        self.mainWindow.comboBox_filterType.currentTextChanged.connect(
            lambda: self.freq_domain_filters(self.mainWindow.radioButton_lowPass.isChecked()
                                             , self.mainWindow.comboBox_filterType.currentText()))
        self.mainWindow.btn_addFirstImage.clicked.connect(self.toggle_data)
        self.mainWindow.btn_addSecondImage.clicked.connect(self.toggle_data)
        self.mainWindow.btn_addFirstImage.clicked.connect(
            lambda: self.freq_domain_filters(True,
                                             self.mainWindow.comboBox_filterType.currentText()))
        self.mainWindow.btn_addSecondImage.clicked.connect(lambda:self.freq_domain_filters(False,
                                                                                self.mainWindow.comboBox_filterType.currentText()))

        self.mainWindow.btn_applyHybrid.clicked.connect(
            lambda:self.mix_images(self.img_data_low_pass,self.img_data_high_pass))
    def updateFrequencyValue(self, value):
        self.mainWindow.label_frequencyValue.setText('Cut-off Frequency: {} Hz'.format(value))
        self.freq_domain_filters( self.mainWindow.radioButton_lowPass.isChecked()
                                 ,  self.mainWindow.comboBox_filterType.currentText())
    def updateTValue(self, value):
        self.mainWindow.label_valueOfT.setText('T-Value: {}'.format(value))    

    def fourier_transform(self,img):
        return np.fft.fftshift(np.fft.fft2(img))

    def calculate_distance(self,row,column,kernal_size):
        return np.sqrt((np.power(row - np.floor(kernal_size[0]/2),2))+
                        (np.power(column - np.floor(kernal_size[1]/2),2)))



    def ideal_filter(self,img_data, cut_off_freq, low_pass_filter):
        kernal = np.zeros((img_data.shape), dtype=np.float32)
        for row in range(kernal.shape[0]):
            for column in range(kernal.shape[1]):
                d = self.calculate_distance(row, column, img_data.shape)
                if d <= cut_off_freq:
                    if low_pass_filter:
                        kernal[row, column] = 1
                    else:
                        kernal[row, column] = 0
                else:
                    if low_pass_filter:
                        kernal[row, column] = 0
                    else:
                        kernal[row, column] = 1
        return self.apply_convolution(img_data, kernal)

    def butterworth_filter(self,img_data, cut_off_freq, low_pass_filter):
        kernal = np.zeros((img_data.shape), dtype=np.float32)
        for row in range(kernal.shape[0]):
            for column in range(kernal.shape[1]):
                d = self.calculate_distance(row, column, img_data.shape)
                kernal_data = 1/(1+((d/cut_off_freq)**2))
                if low_pass_filter:
                    kernal[row, column] = kernal_data
                else:
                    kernal[row, column] = 1 - kernal_data
        return self.apply_convolution(img_data, kernal)

    def gaussian_filter(self,img_data, cut_off_freq, low_pass_filter):
        kernal = np.zeros((img_data.shape), dtype=np.float32)
        for row in range(kernal.shape[0]):
            for column in range(kernal.shape[1]):
                d = self.calculate_distance(row, column, img_data.shape)
                kernal_data = np.exp(-(d**2)/(2*(cut_off_freq**2)))
                if low_pass_filter:
                    kernal[row, column] = kernal_data
                else:
                    kernal[row, column] = 1 - kernal_data
        return self.apply_convolution(img_data, kernal)

    def hyprid_images(self,img_data1, img_data2):
        return 0.5*img_data1 + 0.5*cv2.resize(img_data2, (img_data1.shape[1], img_data1.shape[0]))


    def toggle_data(self):
        self.image_mixed = True
        # self.mainWindow.browse_image()
        self.open_image()

    def open_image(self): 
        if self.image_mixed:
            options = QFileDialog.Options()
            file_name, _ = QFileDialog.getOpenFileName(self.mainWindow, "Select Image", "",
                                                "Image Files (*.png *.jpg *.jpeg *.bmp *.gif *.webp)",
                                                options=options)
            self.image_data = cv2.imread(file_name, 0)
            

        else:
            print(f"shape:{self.mainWindow.selected_image_path}") 
            if self.mainWindow.selected_image_path:
                self.image_data = cv2.imread(self.mainWindow.selected_image_path, 0)
                self.read_image(self.mainWindow.graphicsLayoutWidget_displayImagesMain_,self.image_data)

      


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
        self.vb.setAspectLocked(True)  # Lock aspect ratio to prevent scaling
        self.vb.autoRangeEnabled = False  # Disable auto-scaling

    def freq_domain_filters(self,low_pass_applied,type_filter="Ideal"):
            cut_off_freq = int(self.mainWindow.slider_adjustFrequency.value())
            img_filtered_data = self.filter_types[type_filter](self.image_data, cut_off_freq, low_pass_applied)

            if self.image_mixed:
                self.counter += 1
                if low_pass_applied:
                    self.img_data_low_pass = img_filtered_data
                    self.read_image(self.mainWindow.image_lowPass, np.abs(img_filtered_data))
                else:
                    self.img_data_high_pass = img_filtered_data
                    self.read_image(self.mainWindow.image_highPass, np.abs(img_filtered_data))
            else:
                self.read_image(self.mainWindow.image_filteredFrequencies, np.abs(img_filtered_data))






    def mix_images(self,img_data1,img_data2):
        result = self.hyprid_images(img_data1,img_data2)
        self.read_image(self.mainWindow.image_result, result)

