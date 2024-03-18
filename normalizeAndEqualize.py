import numpy as np
import cv2
import pyqtgraph as pg





# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------





class ImageProcessor:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.ui.eq_normal_combobox.activated.connect(self.imageProcessing)

    def imageProcessing(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            self.ui.image_beforeOperation.clear()
            original_img_item = pg.ImageItem(imageArray)
            original_view = self.ui.image_beforeOperation.addViewBox()
            original_view.addItem(original_img_item)
            self.original_image = imageArray



    def updateDisplay(self, image):
        # Clear existing items in the appropriate view box
        view_box = self.ui.image_edges.addViewBox()
        self.ui.image_edges.clear()
        # Display the new image
        img_item = pg.ImageItem(image)
        view_box.addItem(img_item)