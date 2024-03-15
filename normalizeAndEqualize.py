from PIL import Image
import numpy as np

class ImageEqualizerAndNormalization:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = Image.open(self.image_path)
    
    def histogram_equalization(self):
        # Convert image to grayscale
        gray_img = self.image.convert('L')
        # Calculate histogram
        histogram = np.bincount(np.array(gray_img).flatten(), minlength=256)
        # Compute cumulative distribution function (CDF)
        cdf = histogram.cumsum()
        # Normalize the CDF
        cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
        cdf_normalized = cdf_normalized.astype('uint8')
        # Use the CDF to map the pixel intensities in the original image to new values
        equalized_img = cdf_normalized[np.array(gray_img)]
        # Convert back to an image
        return Image.fromarray(equalized_img)
    
    def normalize_image(self, new_min=0, new_max=255):
        # Convert image to numpy array
        img_array = np.array(self.image)
        # Calculate the min and max pixel values
        old_min = img_array.min()
        old_max = img_array.max()
        # Apply normalization formula
        normalized_array = (img_array - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
        # Clip values to the new_min and new_max
        normalized_array = np.clip(normalized_array, new_min, new_max)
        # Convert back to image
        return Image.fromarray(normalized_array.astype('uint8'))

    def save_image(self, img, output_path):
        img.save(output_path)
    
    def process_and_save_images(self):
        # Process the images
        equalized_image = self.histogram_equalization()
        normalized_image = self.normalize_image()

        # Save the processed images
        self.save_image(equalized_image, 'equalized_image.jpg')
        self.save_image(normalized_image, 'normalized_image.jpg')

# Usage
image_processor = ImageEqualizerAndNormalization('path_to_your_image.jpg')
image_processor.process_and_save_images()
