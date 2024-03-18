import numpy as np
# ----------------



class Thresholding:
    def __init__(self, img):
        self.img = img

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    def adaptive_threshold_gaussian(self, block_size, slider_value):
        assert block_size % 2 == 1 and block_size > 0, "block_size must be an odd positive integer"

        height, width = self.img.shape
        binary = np.zeros((height, width), dtype=np.uint8)

        for i in range(height):
            for j in range(width):
                x_min = max(0, i - block_size // 2)
                y_min = max(0, j - block_size // 2)
                x_max = min(height - 1, i + block_size // 2)
                y_max = min(width - 1, j + block_size // 2)
                block = self.img[x_min:x_max+1, y_min:y_max+1]
                mean = np.mean(block)
                std = np.std(block)
                thresh = mean - slider_value * std
                if self.img[i, j] >= thresh:
                    binary[i, j] = 255
                else:
                    binary[i, j] = 0

        return binary

# -----------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------

    @staticmethod
    def global_thresholding(value, gray_image):
        global_threshold = np.zeros_like(gray_image)
        threshold = value
        global_threshold[gray_image > threshold] = 255
        global_threshold[gray_image <= threshold] = 0
        return global_threshold

# Example usage
gray_image = np.random.randint(0, 256, size=(100, 100)).astype(np.uint8)
thresholding = Thresholding(gray_image)
adaptive_binary = thresholding.adaptive_threshold_gaussian(3, 0.5)
global_binary = Thresholding.global_thresholding(127, gray_image)
