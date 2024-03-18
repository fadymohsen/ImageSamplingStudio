import numpy as np
import cv2
import pyqtgraph as pg





# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------





class EdgeDetector:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        self.ui.comboBox_edgeMaskType.activated.connect(self.detectEdges)
        self.ui.comboBox_edgeMaskDirection.activated.connect(self.detectEdges)

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
    
    def detectEdges(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            self.ui.image_beforeEdgeDetection.clear()
            original_img_item = pg.ImageItem(imageArray)
            original_view = self.ui.image_beforeEdgeDetection.addViewBox()
            original_view.addItem(original_img_item)
            self.original_image = imageArray


            selected_mask_type = self.ui.comboBox_edgeMaskType.currentText()
            selected_direction = self.ui.comboBox_edgeMaskDirection.currentText()

            if selected_mask_type == "Sobel":
                edges_image = self.sobelEdgeDetection(imageArray, selected_direction)
            elif selected_mask_type == "Prewitt":
                edges_image = self.prewittEdgeDetection(imageArray, selected_direction)
            elif selected_mask_type == "Robert":
                edges_image = self.robertEdgeDetection(imageArray, selected_direction)
            elif selected_mask_type == "Canny":
                edges_image = self.cannyEdgeDetection(imageArray)

            # Add the edges image to image_afterEdgeDetection
            self.ui.image_afterEdgeDetection.clear()
            edges_img_item = pg.ImageItem(edges_image)
            edges_view = self.ui.image_afterEdgeDetection.addViewBox()
            edges_view.addItem(edges_img_item)

            self.updateDisplay(edges_image)


# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def updateDisplay(self, image):
        # Clear existing items in the appropriate view box
        view_box = self.ui.image_edges.addViewBox()
        self.ui.image_edges.clear()
        # Display the new image
        img_item = pg.ImageItem(image)
        view_box.addItem(img_item)

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def sobelEdgeDetection(self, image, direction):
        print("sobelEdgeDetection Is Selected")
        if direction == "Vertical":
            sobelFilter = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]])
        else:
            sobelFilter = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])
        
        sobelEdgeDetectedImage = cv2.filter2D(image, -1, sobelFilter)
        return sobelEdgeDetectedImage

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def prewittEdgeDetection(self, image, direction):
        print("prewittEdgeDetection Is Selected")
        if direction == "Vertical":
            prewittFilter = np.array([[-1, 0, 1],
                                    [-1, 0, 1],
                                    [-1, 0, 1]])
        else:
            prewittFilter = np.array([[1, 1, 1],
                                    [0, 0, 0],
                                    [-1, -1, -1]])
        prewittEdgeDetectedImage = cv2.filter2D(image, -1, prewittFilter)
        return prewittEdgeDetectedImage

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def robertEdgeDetection(self, image, direction):
        print("robertEdgeDetection Is Selected")
        if direction == "Vertical":
            robertFilter = np.array([[1, 0],
                                    [0, -1]])
        else:
            robertFilter = np.array([[0, 1],
                                    [-1, 0]])
        robertEdgeDetectedImage = cv2.filter2D(image, -1, robertFilter)
        return robertEdgeDetectedImage

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def cannyEdgeDetection(self, image):
        print("CannyEdgeDetection Is Selected")
        blurred_img = cv2.GaussianBlur(image, (5, 5), 4)

        sobelFilterVertical = np.array([[-1, 0, 1],
                                        [-2, 0, 2],
                                        [-1, 0, 1]])

        sobelFilterHorizontal = np.array([[-1, -2, -1],
                                            [0, 0, 0],
                                            [1, 2, 1]])

        Gradient_x = cv2.filter2D(blurred_img, -1, sobelFilterVertical) 
        Gradient_y = cv2.filter2D(blurred_img, -1, sobelFilterHorizontal) 

        Gradient = np.hypot(Gradient_x, Gradient_y)
        Gradient = Gradient / Gradient.max() * 255
        angle = np.arctan2(Gradient_y, Gradient_x)

        M, N = Gradient.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = angle * 180 / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                q = 255
                r = 255
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    r = Gradient[i, j - 1]
                    q = Gradient[i, j + 1]
                elif (22.5 <= angle[i, j] < 67.5):
                    r = Gradient[i - 1, j + 1]
                    q = Gradient[i + 1, j - 1]
                elif (67.5 <= angle[i, j] < 112.5):
                    r = Gradient[i - 1, j]
                    q = Gradient[i + 1, j]
                elif (112.5 <= angle[i, j] < 157.5):
                    r = Gradient[i + 1, j + 1]
                    q = Gradient[i - 1, j - 1]

                if (Gradient[i, j] >= q) and (Gradient[i, j] >= r):
                    Z[i, j] = Gradient[i, j]
                else:
                    Z[i, j] = 0
        thinEdgesImage = Z

        lowThresholdRatio = 0.05
        highThresholdRatio = 0.09
        highThreshold = thinEdgesImage.max() * highThresholdRatio
        lowThreshold = highThreshold * lowThresholdRatio
        M, N = Z.shape
        thresholdedImage = np.zeros((M, N), dtype=np.int32)
        weak = 153
        strong = 255

        strong_i, strong_j = np.where(thinEdgesImage >= highThreshold)
        weak_i, weak_j = np.where((thinEdgesImage <= highThreshold) & (thinEdgesImage >= lowThreshold))
        thresholdedImage[strong_i, strong_j] = strong
        thresholdedImage[weak_i, weak_j] = weak

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if thresholdedImage[i, j] == weak:
                    if (
                        (thresholdedImage[i + 1, j - 1] == strong) or (thresholdedImage[i + 1, j] == strong) or
                        (thresholdedImage[i + 1, j + 1] == strong) or (thresholdedImage[i, j - 1] == strong) or
                        (thresholdedImage[i, j + 1] == strong) or (thresholdedImage[i - 1, j - 1] == strong) or
                        (thresholdedImage[i - 1, j] == strong) or (thresholdedImage[i - 1, j + 1] == strong)
                    ):
                        thresholdedImage[i, j] = strong
                    else:
                        thresholdedImage[i, j] = 0
        return thresholdedImage
