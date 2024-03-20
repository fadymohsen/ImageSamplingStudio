import numpy as np
import pyqtgraph as pg
import cv2
# -------------------------



class Thresholding:
    def __init__(self, MainWindow):
        self.img = None
        self.global_threshold = None
        self.ui=MainWindow


    def handleButton1(self):    
        self.readImage()
        self.ui.slider_adjustTValue.setRange(0, 255)  
        self.ui.slider_adjustTValue.setValue(127)  
        self.ui.slider_adjustTValue.sliderReleased.connect(self.updateThreshold)
        self.ui.radioButton_globalThresholding.toggled.connect(self.displayImagesThreshold)
        self.ui.radioButton_localThresholding.toggled.connect(self.displayImagesThreshold)
        

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
    def readImage(self):
        self.img =  cv2.imread(self.ui.selected_image_path,0)
        self.img= np.rot90(self.img, -1)
         
    def adaptive_thresholdGaussian(self,gray_image, block_size, slider_value):
        
        assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

        height, width = gray_image.shape
        binary = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                x_min = max(0, i - block_size // 2)
                y_min = max(0, j - block_size // 2)
                x_max = min(height - 1, i + block_size // 2)
                y_max = min(width - 1, j + block_size // 2)
                block = gray_image[x_min:x_max+1, y_min:y_max+1]
                mean = np.mean(block)
                std = np.std(block)
                thresh = mean - slider_value * std
                if gray_image[i, j] >= thresh:
                    binary[i, j] = 255
                else:
                    binary[i, j] = 0

        return binary

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    
    def GlobalThresholding(self,value, gray_image):
        global_threshold = np.zeros_like(gray_image)
        threshold = value
        global_threshold[gray_image > threshold] = 255
        global_threshold[gray_image <= threshold] = 0
        return global_threshold

    def updateThreshold(self):
        value=int(self.ui.slider_adjustTValue.value())
        gray_image = self.img
        self.global_threshold = self.GlobalThresholding(value,gray_image)

        # Local thresholding
        block_size = 39
        self.local_threshold = self.adaptive_thresholdGaussian(gray_image, block_size, value/255)
        self.displayImagesThreshold()



    def displayImagesThreshold(self):
        gray_image = self.img
        self.ui.image_beforeThresholding.clear()
        ImageBeforeThresholding = pg.ImageItem(gray_image)
        BeforeThresholding = self.ui.image_beforeThresholding.addViewBox()
        BeforeThresholding.addItem(ImageBeforeThresholding)

        
        if self.ui.radioButton_globalThresholding.isChecked():
            # Perform global thresholding
            self.ui.image_afterThresholding.clear()
            ImageAfterThresholding = pg.ImageItem(self.global_threshold)
            AfterThresholding = self.ui.image_afterThresholding.addViewBox()
            AfterThresholding.addItem(ImageAfterThresholding)
            

        elif self.ui.radioButton_localThresholding.isChecked():
            # Perform local thresholding
            self.ui.image_afterThresholding.clear()
            ImageAfterThresholding = pg.ImageItem(self.local_threshold)
            AfterThresholding = self.ui.image_afterThresholding.addViewBox()
            AfterThresholding.addItem(ImageAfterThresholding)    
            

        else:
            # Neither radio button is checked
            self.ui.image_afterThresholding.clear()