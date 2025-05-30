# Import required libraries
import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt


class ImageSegmenter:
    """
    A class for segmenting images using a pre-trained machine learning model.
    Handles image cropping, feature extraction, segmentation, and result saving.
    """
    
    def __init__(self):
        # Initialize paths and parameters
        self.new_dir = r'S:\Frank\Wheat_exp_drought\Analysis\side-V\Segmented_side\\DS-HN-CC'
        self.dir_org = r'S:\Frank\Wheat_exp_drought\Analysis\side-V\CC-done'
        
        # Crop parameters: [(y_start, x_start), (height, width)]
        self.crop_params = [(800, 1000), (3500, 4000)]  
        
        # Load segmentation model
        self.segmentation_model = self._load_model('MLP_new_26.sav')
        
    def _load_model(self, model_path):
        """Load pre-trained segmentation model from file."""
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
    def _extract_features(self, image):
        """
        Extract features from image in multiple color spaces.
        
        Args:
            image: Input image in BGR format
            
        Returns:
            numpy.ndarray: Array of extracted features for each pixel
        """
        # Convert to different color spaces
        rgb_float = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_rgb = image.reshape(-1, 3)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv_float = cv2.normalize(hsv, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hsv_float = hsv_float.reshape(-1, 3)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab_float = cv2.normalize(lab, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lab_float = lab_float.reshape(-1, 3)

        # Combine features from all color spaces
        features = []
        for i in range(img_rgb.shape[0]):
            pixel_features = np.hstack([
                img_rgb[i, 0], img_rgb[i, 1], img_rgb[i, 2],
                hsv_float[i, 0], hsv_float[i, 1], hsv_float[i, 2],
                lab_float[i, 0], lab_float[i, 1], lab_float[i, 2]
            ])
            features.append(pixel_features)
            
        return np.array(features)
    
    def _process_image(self, image_path):
        """
        Process a single image through the segmentation pipeline.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            tuple: (original_cropped_image, segmented_image)
        """
        # Read and crop image
        print(f'[INFO] Processing image: {image_path}')
        img = cv2.imread(image_path)
        if img is None:
            print(f'[WARNING] Could not read image: {image_path}')
            return None, None
            
        y, x = self.crop_params[0]
        h, w = self.crop_params[1]
        cropped_img = img[y:y + h, x:x + w]
        
        # Extract features and predict
        features = self._extract_features(cropped_img)
        pixel_predict = self.segmentation_model.predict(features)
        
        # Create mask and apply to image
        img_mask = pixel_predict.reshape((cropped_img.shape[0], cropped_img.shape[1])).astype("uint8")
        img_mask = cv2.medianBlur(img_mask, 3)  # Remove small noise
        
        # Apply mask to original image
        segmented_img = np.ones(cropped_img.shape, dtype="uint8") * 255
        segmented_img[img_mask != 0] = cropped_img[img_mask != 0]
        
        return cropped_img, segmented_img
    
    def process_directory(self):
        """Process all images in the input directory and save results."""
        # Verify output directory exists
        os.makedirs(self.new_dir, exist_ok=True)
        
        # Process each image in directory
        for root, _, files in os.walk(self.dir_org):
            for filename in files:
                if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                    
                input_path = os.path.join(root, filename)
                _, segmented_img = self._process_image(input_path)
                
                if segmented_img is not None:
                    # Convert to RGB and save
                    output_img = cv2.cvtColor(segmented_img, cv2.COLOR_BGR2RGB)
                    output_path = os.path.join(self.new_dir, filename)
                    cv2.imwrite(output_path, output_img)
                    print(f'[INFO] Saved segmented image: {output_path}')


if __name__ == "__main__":
    try:
        segmenter = ImageSegmenter()
        segmenter.process_directory()
        print("[INFO] Image segmentation completed successfully")
    except Exception as e:
        print(f"[ERROR] Segmentation failed: {e}")
