import numpy as np
import pyqtgraph as pg
import cv2


class RGBHistograms:
    def __init__(self, MainWindow):
        self.Ui = MainWindow
        self.image=None
        
    def handleButton(self):
        self.readImage()
        self.Ui.radioButton_normalHistogram.toggled.connect(self.drawHistograms)
        self.Ui.radioButton_cumulative.toggled.connect(self.drawHistograms)
        
    def readImage(self):
        self.image =  cv2.imread(self.Ui.selected_image_path)

    def drawRGBHistograms(self,image):
        # Initialize histograms for each channel
        histogram_r = np.zeros(256, dtype=np.uint)
        histogram_g = np.zeros(256, dtype=np.uint)
        histogram_b = np.zeros(256, dtype=np.uint)
        cdf_r = np.zeros(256, dtype=np.uint)
        cdf_g = np.zeros(256, dtype=np.uint)
        cdf_b = np.zeros(256, dtype=np.uint)
        cumsum_r = 0
        cumsum_g = 0
        cumsum_b = 0

        # Iterate over each pixel in the image
        for row in image:
            for pixel in row:
                # Extract RGB values for the pixel
                r, g, b = pixel

                # Increment the corresponding histogram bin
                histogram_r[r] += 1
                histogram_g[g] += 1
                histogram_b[b] += 1

        # Calculate cumulative sums for each channel
        for i in range(256):
            cumsum_r += histogram_r[i]
            cumsum_g += histogram_g[i]
            cumsum_b += histogram_b[i]

            # Update cumulative functions
            cdf_r[i] = cumsum_r
            cdf_g[i] = cumsum_g
            cdf_b[i] = cumsum_b

        return histogram_r,histogram_g,histogram_b,cdf_r,cdf_g,cdf_b

    def drawHistograms(self):
        
        histogram_r,histogram_g,histogram_b,cdf_r,cdf_g,cdf_b = self.drawRGBHistograms(self.image)
        
        if self.Ui.radioButton_normalHistogram.isChecked():

            
            # Plot histograms in respective PlotWidget objects
            self.plotHistogram(histogram_r, 'red', self.Ui.image_redHistogram)
            self.plotHistogram(histogram_g, 'green', self.Ui.image_greenHistogram)
            self.plotHistogram(histogram_b, 'blue', self.Ui.image_blueHistogram)
            

        elif self.Ui.radioButton_cumulative.isChecked(): 
            
            
            # Plot cumulative functions
            self.plotCumulative(cdf_r, 'r', self.Ui.image_redHistogram)
            self.plotCumulative(cdf_g, 'g', self.Ui.image_greenHistogram)
            self.plotCumulative(cdf_b, 'b', self.Ui.image_blueHistogram)
            
        else:
            
            # Neither radio button is checked
            self.Ui.image_redHistogram.clear()
            self.Ui.image_greenHistogram.clear()
            self.Ui.image_blueHistogram.clear()
            

    def plotCumulative( self,cdf, color, plot_widget):
        x = np.arange(256)
        plot_widget.clear()
        plot_cum_RGBHistograms =pg.PlotCurveItem(x, cdf, pen=color)  # Plot cumulative histogram
        view_cum_RGBHistograms = plot_widget.addViewBox()
        view_cum_RGBHistograms.addItem(plot_cum_RGBHistograms)

    def plotHistogram( self,histogram, color, plot_widget):
        x = np.arange(256)
        plot_widget.clear()  # Clear existing plot
        plot_RGBHistograms =pg.PlotCurveItem(x, histogram, pen=color)  # Plot histogram
        view_RGBHistograms = plot_widget.addViewBox()
        view_RGBHistograms.addItem(plot_RGBHistograms)
    