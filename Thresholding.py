import numpy as np

def adaptive_thresholdGaussian(img, block_size, slider_value):
    # Check that the block size is odd and nonnegative
    assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

    # Calculate the local threshold for each pixel
    height, width = img.shape
    binary = np.zeros((height, width), dtype=np.uint8)

    for i in range(height):
        for j in range(width):
            # Calculate the local threshold using a square neighborhood centered at (i, j)
            x_min = max(0, i - block_size // 2)
            y_min = max(0, j - block_size // 2)
            x_max = min(height - 1, i + block_size // 2)
            y_max = min(width - 1, j + block_size // 2)
            block = img[x_min:x_max+1, y_min:y_max+1]
            mean = np.mean(block)
            std = np.std(block)
            thresh = mean - slider_value * std
            if img[i, j] >= thresh:
                binary[i, j] = 255
            else:
                binary[i, j] = 0

    return binary

def GlobalThresholding(value,gray_image):
        # Global thresholding
        global_threshold = np.zeros_like(gray_image)
        threshold = value
        global_threshold[gray_image > threshold] = 255
        global_threshold[gray_image <= threshold] = 0
        return global_threshold