import numpy as np
import pyqtgraph as pg

def drawRGBHistograms(image):
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

def plotCumulative( cdf, color, plot_widget):
    x = np.arange(256)
    plot_widget.clear()
    plot_cum_rgbhistograms =pg.PlotCurveItem(x, cdf, pen=color)  # Plot cumulative histogram
    view_cum_rgbhistograms = plot_widget.addViewBox()
    view_cum_rgbhistograms.addItem(plot_cum_rgbhistograms)

def plotHistogram( histogram, color, plot_widget):
    x = np.arange(256)
    plot_widget.clear()  # Clear existing plot
    plot_rgbhistograms =pg.PlotCurveItem(x, histogram, pen=color)  # Plot histogram
    view_rgbhistograms = plot_widget.addViewBox()
    view_rgbhistograms.addItem(plot_rgbhistograms)
    