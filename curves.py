import numpy as np
import pyqtgraph as pg
import cv2





# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------





class Curves:
    def __init__(self, main_tab_widget):
        self.main_tab_widget = main_tab_widget
        self.ui = self.main_tab_widget
        
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------
    
    def drawCurves(self):
        if self.main_tab_widget.selected_image_path:
            imageArray = cv2.imread(self.main_tab_widget.selected_image_path)
            if imageArray.ndim == 3:
                imageArray = cv2.cvtColor(imageArray, cv2.COLOR_BGR2GRAY)
            imageArray = cv2.rotate(imageArray, cv2.ROTATE_90_CLOCKWISE)
            
            histogramCurve, distributionCurve = self.drawHistogram(imageArray)

            view = self.ui.image_histogramCurve.addViewBox()
            view.addItem(histogramCurve)
            distributionView = self.ui.image_distributionCurve.addViewBox()
            distributionView.addItem(distributionCurve )

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def drawHistogram(self, image):
        image = image.flatten()
        histogram = [0] * 50
        for pixel_value in image:
            # Increment the corresponding bin in the histogram
            bin_index = min(pixel_value // 5, 49)  # Map pixel value to bin index
            histogram[bin_index] += 1
        
        x = np.linspace(0, 255, 49)
        y = histogram
        x = np.append(x, 255)
        histogramCurve = pg.BarGraphItem(x = x, height = y, width = 5, brush ='blue') 

        x = np.linspace(0, 255, 50)
        x = np.append(x, 255)
        distributionCurve =  pg.PlotCurveItem(x, y, stepMode=True, fillLevel=0, brush=(0, 0, 255, 0), pen=pg.mkPen(color='r', width=1))
        return histogramCurve, distributionCurve 
        
# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def drawDistributionCurve(self, image, histogram):
        x = np.linspace(0, 255, 50)
        x = np.append(x, 255)
        distributionCurve =  pg.PlotCurveItem(x, histogram, stepMode=True, fillLevel=0, brush=(0, 0, 255, 0), pen=pg.mkPen(color='r', width=1))
        return distributionCurve
        