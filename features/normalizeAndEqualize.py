import numpy as np
import cv2
import pyqtgraph as pg

class ImageProcessor:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.ui.eq_normal_combobox.activated.connect(self.imageProcessing)
        self.original_image = None

    def imageProcessing(self):
        selected_option = self.ui.eq_normal_combobox.currentText()
        image_path = self.main_tab_widget.selected_image_path
        if image_path:
            image_array = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if image_array.ndim == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            image_array = cv2.rotate(image_array, cv2.ROTATE_90_CLOCKWISE)
            self.updateDisplay(image_array, self.ui.image_beforeOperation)
            self.original_image = image_array

            if selected_option == "Normalization":
                processed_image = self.image_normalization(image_array)
            elif selected_option == "Equalization":
                processed_image = self.histogram_equalization(image_array)
            else:
                return

            self.updateDisplay(processed_image, self.ui.image_afterOperation)

    def image_normalization(self, image_array):
        image_float = image_array.astype(np.float32)
        min_val = np.min(image_float)
        max_val = np.max(image_float)
        normalized_image = ((image_float - min_val) / (max_val - min_val)) * 255
        return normalized_image.astype(np.uint8)

    def histogram_equalization(self, image, max_value=255):
        hist = self.get_histogram(image.flatten(), 256)
        cdf = self.get_cdf(hist, image.shape)
        normalize = np.rint(cdf * max_value).astype('int')
        result = normalize[image.flatten()]
        return result.reshape(image.shape)

    def get_histogram(self, image_data, bins):
        histogram = np.zeros(bins)
        for value in image_data:
            histogram[value] += 1
        return histogram

    def get_cdf(self, histogram, shape):
        cdf = histogram.cumsum()
        cdf_normalized = cdf / cdf[-1]
        return cdf_normalized * np.prod(shape)

    def updateDisplay(self, image, image_view_widget):
        image_view_widget.clear()
        view_box = image_view_widget.addViewBox()
        view_box.setAspectLocked(True)
        img_item = pg.ImageItem(image)
        view_box.addItem(img_item)