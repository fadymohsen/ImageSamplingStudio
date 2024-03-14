import numpy as np
import cv2

def sobelEdgeDetection(image, direction):
        if direction == "Vertical":
            sobelFilter = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]] )
        else:
            sobelFilter = np.array([[-1, -2, -1],
                                    [0, 0, 0],
                                    [1, 2, 1]])
        
        # Apply the Sobel filter to the grayscale image
        #The -1 argument indicates that the output image should have the same depth as the input image.
        sobelEdgeDetectedImage = cv2.filter2D(image, -1, sobelFilter) 
        return sobelEdgeDetectedImage

    
def prewittEdgeDetection(image, direction):   
    if direction == "Vertical":
        prewittFilter = np.array([[-1, 0, 1],
                                [-1, 0, 1],
                                [-1, 0, 1]] )
    else:
        prewittFilter = np.array([[1, 1, 1],
                                [0, 0, 0],
                                [-1, -1, -1]])
    prewittEdgeDetectedImage = cv2.filter2D(image, -1, prewittFilter) 
    return prewittEdgeDetectedImage


def robertEdgeDetection(image, direction):
    if direction == "Vertical":
        prewittFilter = np.array([[1, 0],
                                [0, -1]])
    else:
        prewittFilter = np.array([[0, 1],
                                [-1, 0]])
    robertEdgeDetectedImage = cv2.filter2D(image, -1, prewittFilter) 
    return robertEdgeDetectedImage

def cannyEdgeDetection(image):
    #Gaussian filter on the grayscale image to reduce noise
    blurred_img = cv2.GaussianBlur(image, (5, 5), 4)

    #Calculating the gradients and its directions using sobel filter
    sobelFilterVertical = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]] )

    sobelFilterHorizontal = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

    Gradient_x = cv2.filter2D(blurred_img, -1, sobelFilterVertical) 
    Gradient_y = cv2.filter2D(blurred_img, -1, sobelFilterHorizontal) 

    # Gradient = np.sqrt(np.square(Gradient_x) + np.square(Gradient_y))
    
    Gradient = np.hypot(Gradient_x, Gradient_y)

    Gradient = Gradient / Gradient.max() * 255
    
    angle = np.arctan2(Gradient_y, Gradient_x)

    #Non Maximum Suppression to reduce the width of the edges
    M, N = Gradient.shape
    Z = np.zeros((M,N), dtype=np.int32) # resultant image
    
    angle = angle * 180 / np.pi        # max -> 180, min -> -180
    angle[angle < 0] += 180             # max -> 180, min -> 0

    for i in range(1,M-1):
        for j in range(1,N-1):
            
            #Initializing the adjacent pixels for the current pixel
            q = 255
            r = 255
        
            #Checking the direction of the current pixel to determine it's adjacent pixels
            
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                r = Gradient[i, j-1]
                q = Gradient[i, j+1]
            
            elif (22.5 <= angle[i,j] < 67.5):
                r = Gradient[i-1, j+1]
                q = Gradient[i+1, j-1]
            
            elif (67.5 <= angle[i,j] < 112.5):
                r = Gradient[i-1, j]
                q = Gradient[i+1, j]
            
            elif (112.5 <= angle[i,j] < 157.5):
                r = Gradient[i+1, j+1]
                q = Gradient[i-1, j-1]

            #Checking wether the current pixel is larger than its adjacents or not
            if (Gradient[i,j] >= q) and (Gradient[i,j] >= r):
                Z[i,j] = Gradient[i,j]
            else:
                Z[i,j] = 0
    thinEdgesImage = Z


    #Double Thresholding and Hystresis to determine wether the pixel is strong, weak or non-edge
                
    lowThresholdRatio=0.05
    highThresholdRatio=0.09

    highThreshold = thinEdgesImage.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = Z.shape
    thresholdedImage = np.zeros((M,N), dtype=np.int32)

    weak = 153
    strong = 255
    
    #Finding the indices for strong weak and non-edge pixels
    strong_i, strong_j = np.where(thinEdgesImage >= highThreshold)
    weak_i, weak_j = np.where((thinEdgesImage <= highThreshold) & (thinEdgesImage>= lowThreshold))

    thresholdedImage[strong_i, strong_j] = strong
    thresholdedImage[weak_i, weak_j] = weak


    #Hystresis process to detemrine whether the weak edges are considered edges or non-edges
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (thresholdedImage[i, j] == weak):
                if (
                    (thresholdedImage[i+1, j-1] == strong) or (thresholdedImage[i+1, j] == strong) or
                    (thresholdedImage[i+1, j+1] == strong) or (thresholdedImage[i, j-1] == strong) or
                    (thresholdedImage[i, j+1] == strong) or (thresholdedImage[i-1, j-1] == strong) or
                    (thresholdedImage[i-1, j] == strong) or (thresholdedImage[i-1, j+1] == strong)
                ):
                    thresholdedImage[i, j] = strong
                else:
                    thresholdedImage[i, j] = 0
    return thresholdedImage