import numpy as np
import pyqtgraph as pg


def drawCurves(image):
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
       

def drawDistributionCurve(image, histogram):
    x = np.linspace(0, 255, 50)
    x = np.append(x, 255)
    distributionCurve =  pg.PlotCurveItem(x, histogram, stepMode=True, fillLevel=0, brush=(0, 0, 255, 0), pen=pg.mkPen(color='r', width=1))
    return distributionCurve
    