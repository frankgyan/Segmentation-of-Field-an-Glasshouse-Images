# Segmentation-of-Field-an-Glasshouse-Images
Image segmentation is a fundamental but critical step for achieving automated high-throughput phenotyping. 
While conventional segmentation methods perform well in homogenous environments, the performance decreases when used in more complex environments. 
This study aimed to develop a fast and robust neural-network-based segmentation tool to phenotype plants in both field and glasshouse environments in a high-throughput manner. 
Digital images of cowpea (from glasshouse) and wheat (from field) with different nutrient supplies across their full growth cycle were acquired. Image patches from 20 randomly selected images from the acquired dataset were transformed from their original RGB format to multiple color spaces. 
The pixels in the patches were annotated as foreground and background with a pixel having a feature vector of 24 color properties. 
A feature selection technique was applied to choose the sensitive features, which were used to train a multilayer perceptron network (MLP) and two other traditional machine learning models: support vector machines (SVMs) and random forest (RF).
The performance of these models, together with two standard color-index segmentation techniques (excess green (ExG) and excess green–red (ExGR)), was compared. The proposed method outperformed the other methods in producing quality segmented images with over 98%-pixel classification accuracy. 
Regression models developed from the different segmentation methods to predict Soil Plant Analysis Development (SPAD) values of cowpea and wheat showed that images from the proposed MLP method produced models with high 
