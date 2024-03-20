import cv2
import numpy as np
import pyqtgraph as pg
from scipy import signal





# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------





class noiseAdditionFiltration():
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.ui.comboBox_noiseTypes.activated.connect(self.applyNoise)
        self.ui.comboBox_noiseFilters.activated.connect(self.applyFilter)
        self.applyNoise()

    def applyNoise(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            self.ui.image_noiseBeforeEditing.clear()
            original_img_item = pg.ImageItem(imageArray)
            original_view = self.ui.image_noiseBeforeEditing.addViewBox()
            original_view.addItem(original_img_item)
            self.original_image = imageArray
            
            selected_noise = self.ui.comboBox_noiseTypes.currentText()
            if selected_noise == "Uniform Noise":
                self.noisy_image = self.noiseAddition(selected_noise, self.original_image)
                self.updateDisplay(self.noisy_image, is_noisy_image=True)
            elif selected_noise == "Gaussian Noise":
                self.noisy_image = self.noiseAddition(selected_noise, self.original_image)
                self.updateDisplay(self.noisy_image, is_noisy_image=True)
            elif selected_noise == "Salt & Pepper Noise":
                self.noisy_image = self.noiseAddition(selected_noise, self.original_image)
                self.updateDisplay(self.noisy_image, is_noisy_image=True)# Display noisy image on image_noiseAfterEditing
            
            

    def updateDisplay(self, image, is_noisy_image=False):
        # Clear existing items in the appropriate view box
        if is_noisy_image:
            self.ui.image_noiseAfterEditing.clear()
            view_box = self.ui.image_noiseAfterEditing.addViewBox()
        else:
            self.ui.image_noiseBeforeEditing.clear()
            view_box = self.ui.image_noiseBeforeEditing.addViewBox()
        
        # Display the new image
        img_item = pg.ImageItem(image)
        view_box.addItem(img_item)


    def noiseAddition(self, noiseType, image):
        noisyImage = image.copy()  # Create a copy of the original image to avoid accumulation
        if noiseType == "Uniform Noise":
            if noisyImage.ndim == 2:  # For grayscale images
                row, column = noisyImage.shape
                minNum = 0
                maxNum = 0.2
                noise = np.random.uniform(minNum, maxNum, (row, column))
                # Scale the noise to match the image intensity range
                noise *= 255  # Assuming image intensity range is [0, 255]
                # Add noise to the image
                noisyImage = np.clip(noisyImage + noise, 0, 255).astype(np.uint8)
                return noisyImage
            
        elif noiseType == "Gaussian Noise":
            if noisyImage.ndim == 2:
                row, column = noisyImage.shape
                mean = 0
                var = 0.01  # Adjust variance as per requirement
                sigma = np.sqrt(var)
                noise = np.random.normal(mean, sigma, (row, column))
                # Scale the noise to match the image intensity range
                noise *= 255  # Assuming image intensity range is [0, 255]
                # Add noise to the image
                noisyImage = np.clip(noisyImage + noise, 0, 255).astype(np.uint8)
                return noisyImage
            
        elif noiseType == "Salt & Pepper Noise":
            if noisyImage.ndim == 2:
                row, column = noisyImage.shape
                number_of_pixels = np.random.randint(300, 10000)
                for i in range(number_of_pixels):
                    y_coord = np.random.randint(0, row - 1)
                    x_coord = np.random.randint(0, column - 1)
                    noisyImage[y_coord][x_coord] = 255
                number_of_pixels = np.random.randint(300, 10000)
                for i in range(number_of_pixels):
                    y_coord = np.random.randint(0, row - 1)
                    x_coord = np.random.randint(0, column - 1)
                    noisyImage[y_coord][x_coord] = 0
                return noisyImage
            

    def applyFilter(self):
        selected_filter = self.ui.comboBox_noiseFilters.currentText()
        selected_noise = self.ui.comboBox_noiseTypes.currentText()
        self.noisy_image = self.noiseAddition(selected_noise, self.original_image)  # Store the noisy image
        if selected_filter == "Average Filter":
            filtered_image = self.average_filter(self.noisy_image)
            self.updateDisplay(filtered_image, is_noisy_image=True)
        elif selected_filter == "Median Filter":
            filtered_image = self.median_filter(self.noisy_image)
            self.updateDisplay(filtered_image, is_noisy_image=True)
        elif selected_filter == "Gaussian Filter":
            filtered_image = self.gaussian_filter(self.noisy_image)
            self.updateDisplay(filtered_image, is_noisy_image=True)
        


    def average_filter(self, image_data, filter_size=3):
        filter = np.ones((filter_size, filter_size)) / (filter_size ** 2)
        new_img = signal.convolve2d(image_data, filter, mode='same', boundary='fill', fillvalue=0)
        return new_img


    def median_filter(self, image_data, filter_size=3):
        new_img = np.zeros_like(image_data)
        for i in range(image_data.shape[0]):
            for j in range(image_data.shape[1]):
                new_img[i, j] = np.median(image_data[max(0, i-filter_size//2):min(image_data.shape[0], i+filter_size//2+1),
                                                    max(0, j-filter_size//2):min(image_data.shape[1], j+filter_size//2+1)])
        return new_img
 

    def gaussian_filter(self, image, kernel_size=3, sigma=2):
        kernel = self.gaussianKernel(kernel_size, sigma)
        new_img = signal.convolve2d(image, kernel, mode='same', boundary='fill', fillvalue=0)
        return new_img 


    def gaussianKernel(self, kernel_size=3, sig=2):
        ax = np.linspace(-(kernel_size - 1) / 2., (kernel_size - 1) / 2., kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))
        return kernel / np.sum(kernel)