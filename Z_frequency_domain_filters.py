import cv2
import numpy as np

 # define filters 0 -> ideal 1-> butterworth , 2-> gaussian
 #
 # type_filters = {0:
 # ,1:,2:}

def apply_convolution(img, kernel):
    img = fourier_transform(img)
    img_after_kernal = img * kernel
    img_after_kernal = np.fft.ifft2(np.fft.ifftshift(img_after_kernal))
    img_after_kernal = np.clip(img_after_kernal, 0,255)
    return img_after_kernal.astype(np.uint8)

def fourier_transform(img):
    return np.fft.fftshift(np.fft.fft2(img))

def calculate_distance(row,column,kernal_size):
    return np.sqrt((np.power(row - np.floor(kernal_size[0]/2),2))+
                    (np.power(column - np.floor(kernal_size[1]/2),2)))



def ideal_filter(img_data, cut_off_freq, low_pass_filter):
    kernal = np.zeros((img_data.shape), dtype=np.float32)
    for row in range(kernal.shape[0]):
        for column in range(kernal.shape[1]):
            d = calculate_distance(row, column, img_data.shape)
            if d <= cut_off_freq:
                if low_pass_filter:
                    kernal[row, column] = 1
                else:
                    kernal[row, column] = 0
            else:
                if low_pass_filter:
                    kernal[row, column] = 0
                else:
                    kernal[row, column] = 1
    return apply_convolution(img_data, kernal)

def butterworth_filter(img_data, cut_off_freq, low_pass_filter):
    kernal = np.zeros((img_data.shape), dtype=np.float32)
    for row in range(kernal.shape[0]):
        for column in range(kernal.shape[1]):
            d = calculate_distance(row, column, img_data.shape)
            kernal_data = 1/(1+((d/cut_off_freq)**2))
            if low_pass_filter:
                kernal[row, column] = kernal_data
            else:
                kernal[row, column] = 1 - kernal_data
    return apply_convolution(img_data, kernal)

def gaussian_filter(img_data, cut_off_freq, low_pass_filter):
    kernal = np.zeros((img_data.shape), dtype=np.float32)
    for row in range(kernal.shape[0]):
        for column in range(kernal.shape[1]):
            d = calculate_distance(row, column, img_data.shape)
            kernal_data = np.exp(-(d**2)/(2*(cut_off_freq**2)))
            if low_pass_filter:
                kernal[row, column] = kernal_data
            else:
                kernal[row, column] = 1 - kernal_data
    return apply_convolution(img_data, kernal)

def hyprid_images(img_data1, img_data2):
    return 0.5*img_data1 + 0.5*cv2.resize(img_data2, (img_data1.shape[1], img_data1.shape[0]))


