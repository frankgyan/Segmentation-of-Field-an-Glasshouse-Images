
# Import libraries and check their versions
import sys
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import h5py
import pandas as pd
import json
from PIL import Image

# Print library versions for reproducibility
print('Python: {}'.format(sys.version))
print('os: {}'.format(os.version))
print('cv2: {}'.format(cv2.__version__))
print('np: {}'.format(np.__version__))
print('matplotlib: {}'.format(plt.__version__))
print('h5py: {}'.format(h5py.__version__))
print('pandas: {}'.format(pd.__version__))
print('json: {}'.format(json.__version__))
print('PIL: {}'.format(Image.__version__)))


def extract_pixels_from_annotated_patches():
    """
    Main function to extract pixels from annotated patches and label them.
    Processes images from a directory with corresponding JSON annotations.
    """
    
    # Directory containing the stored images
    IMAGE_DIR = 'F:\\New_process\\trial_35_35_only' # Change this to reflect your directory
    
    # Load annotation data from JSON file
    via = {}
    with open('trial_new_35_35.json', 'r') as f:
        via = json.load(f)

    final_data = []
    
    # Process each image in the annotation metadata
    for fid in via['_via_img_metadata']:
        # Construct full file path
        file_path = os.path.join(IMAGE_DIR, via['_via_img_metadata'][fid]['filename'])
        
        # Verify file exists
        if not os.path.isfile(file_path):
            print('File not found! %s' % (file_path))
            continue
            
        # Open image and get dimensions
        im = Image.open(file_path)
        im_width, im_height = im.size
        region_index = 0

        # Process each region in the image
        for region in via['_via_img_metadata'][fid]['regions']:
            # Currently only supports rectangular regions
            if region['shape_attributes']['name'] != 'rect':
                print('Extraction of %s regions not yet implemented!' % 
                      region['shape_attributes']['name'])
                continue
                
            # Get region coordinates
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            w = region['shape_attributes']['width']
            h = region['shape_attributes']['height']

            # Calculate crop boundaries with safety checks
            left = max(0, x)
            top = max(0, y)
            right = min(im_width, x + w)
            bottom = min(im_height, y + h)
            
            # Crop the region of interest
            crop = im.crop((left, top, right, bottom))
            region_index += 1
            
            # Generate unique name for the cropped region
            old_ext = os.path.splitext(via['_via_img_metadata'][fid]['filename'])[1]
            new_ext = old_ext.replace('.', '_' + str(region_index) + '.')
            crop_name = via['_via_img_metadata'][fid]['filename'].replace(old_ext, new_ext)

            # Convert image to various color spaces
            # BGR to RGB
            crop = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2RGB)
            img_rgb = crop.reshape(-1, 3)

            # BGR to HSV
            hsv = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2HSV)
            img_hsv = hsv.reshape(-1, 3)

            # BGR to LAB
            lab = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2LAB)
            img_lab = lab.reshape(-1, 3)

            # BGR to HLS
            hsi = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2HLS)
            img_hsi = hsi.reshape(-1, 3)

            # BGR to Luv color space
            luv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2LUV)
            img_luv = luv.reshape(-1, 3)

            # RGB to YCrCb
            ybr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2YCrCb)
            img_ybr = luv.reshape(-1, 3)

            # RGB to YUV
            yuv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2YUV)
            img_yuv = yuv.reshape(-1, 3)

            # Calculate normalized RGB chromatic coordinates
            R, G, B = cv2.split(crop)
            r = R / (R + G + B)
            g = G / (R + G + B)
            b = B / (R + G + B)
            
            hh = np.dstack((b, g, r))
            nx, ny, ns = hh.shape
            final_n = hh.reshape(nx * ny, ns)
            out_images = np.array(final_n.astype(int))
            final_nRGB = out_images.astype(np.uint8)

            # Extract and combine features from all color spaces
            data_attach = []
            for f in range(img_rgb.shape[0] and img_hsv.shape[0] and img_lab.shape[0] and 
                          img_hsi.shape[0] and img_luv.shape[0] and img_ybr.shape[0] and 
                          img_yuv.shape[0] and final_nRGB.shape[0]):
                
                # Combine features from all color spaces
                data = np.hstack([
                    img_rgb[f, 0], img_rgb[f, 1], img_rgb[f, 2],
                    img_hsv[f, 0], img_hsv[f, 1], img_hsv[f, 2],
                    img_lab[f, 0], img_lab[f, 1], img_lab[f, 2],
                    img_hsi[f, 0], img_hsi[f, 1], img_hsi[f, 2],
                    img_luv[f, 0], img_luv[f, 1], img_luv[f, 2],
                    img_ybr[f, 0], img_ybr[f, 1], img_ybr[f, 2],
                    img_yuv[f, 0], img_yuv[f, 1], img_yuv[f, 2],
                    final_nRGB[f, 0], final_nRGB[f, 1], final_nRGB[f, 2]
                ])

                # Get label from region attributes
                labels = str(region['region_attributes'])
                label_text = labels[12:16]

                # Assign binary label (1 for foreground, 0 for background)
                if label_text == "fore":
                    data = np.append(data, int(1))
                elif label_text == "back":
                    data = np.append(data, int(0))

                data_attach.append(data)
                
            final_data.extend((data_attach))

    # Convert to pandas DataFrame
    feature_df = pd.DataFrame(final_data)
    
    # Store data in HDF5 file
    with h5py.File('cowpea_200_only', 'w') as db:
        dataset = db.create_dataset('datasetName', data=feature_df, dtype="float")


if __name__ == "__main__":
    extract_pixels_from_annotated_patches()
