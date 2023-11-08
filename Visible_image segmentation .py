#== Import python libraries 
import os
import cv2
import pickle
import numpy as np
from matplotlib import pyplot as plt

#==Root to save segmented image 
new_dir= r'S:\Frank\Wheat_exp_drought\Analysis\side-V\Segmented_side\\DS-HN-CC'
#==input desired size for cropping 
P = [(800, 1000) (3500, 4000)]
#==Directory to original file
dir_org=r'S:\Frank\Wheat_exp_drought\Analysis\side-V\CC-done'

#Load segmentation model 
Segmented_file = 'MLP_new_26.sav'
loaded_model = pickle.load(open(Segmented_file, 'rb'))

def croppig(path):
    global features_LAB, features_texture, dissimilarity, contrast, homogeneity, energy, correlation
    # creating empty list to store final extracted features
    final_features = []
    for root, directories, files in os.walk(path, topdown=False):
        for name in files:
            print('[INFO] reading image ' + os.path.join(root, name))
            # Read image data
            img = cv2.imread(os.path.join(root, name))
            #===cropping for rothamsted_cowpea samples
            y = P[0][0]
            x = P[0][1]
            h = P[1][0]
            w = P[1][1]
            visImg = img[y:y + h, x:x + w]
            #==File reshape and stacking to for 1D array
            img_bgr = visImg.reshape(-1, 3)
            img_data = np.vstack([img_bgr[:, 0], img_bgr[:, 1], img_bgr[:, 2]])
            img_data = img_data.transpose()


            #===Extracting features from different colour spaces(LAB, HSV and RGB)
            final_data = []     
            img = visImg
            rgb_float = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            img_rgb = img.reshape(-1, 3)

            hsv = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
            hsv_float = cv2.normalize(hsv, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            hsv_float = hsv_float.reshape(-1, 3)

            lab = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2HSV)
            lab_float = cv2.normalize(lab, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            lab_float = lab_float.reshape(-1, 3)
            data_attach = []
            for i in range(img_rgb.shape[0]):
                img_data = np.hstack([img_rgb[i, 0], img_rgb[i, 1], img_rgb[i, 2],
                                      hsv_float[i, 0], hsv_float[i, 1], hsv_float[i, 2],
                                      lab_float[i, 0], lab_float[i, 1], lab_float[i, 2]])
                data_attach.append(img_data)

            # === predict pixel label using the segmentation model
            pixelPredict = loaded_model.predict(img_data)
            img_mask = pixelPredict.reshape(
                (visImg.shape[0], visImg.shape[1])).astype("uint8")
            img_mask = cv2.medianBlur(img_mask,3)
            # === initialise a white image to overwrite the result
            img_final = np.ones(visImg.shape, dtype="uint8") * 255
            # === mask the result. keep the original pixel values if it is not zero
            img_final[img_mask != 0] = visImg[img_mask != 0]
            # c
            # plt.show()
            #=== convert segmented image back to RGB format
            visImg2 = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            #===Save segmentated data to a directory
            status = cv2.imwrite(new_dir + name, img_final)
path = dir_org
