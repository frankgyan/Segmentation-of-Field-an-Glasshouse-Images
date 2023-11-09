#===import libraries
import cv2
import h5py
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score, RepeatedStratifiedKFold, cross_validate, \
    train_test_split, GridSearchCV
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, plot_confusion_matrix
import pickle
from PIL import Image
import os
import mkl
import json
from sklearn.tree import DecisionTreeClassifier

#***************EXTRACT PIXELS FROM ANNOTATED PATCHES AND LABEL**************************************

#===directory for the stored images
E = 'F:\\New_process\\trial_35_35_only'

#===opening reading json file and corresponding to the image bounding boxes
via = {}
with open('trial_new_35_35.json', 'r') as f:
    via = json.load(f)

#===Extract pixels from the annotated patches 
final_data= []
for fid in via['_via_img_metadata']:
    fn = os.path.join(E, via['_via_img_metadata'][fid]['filename'])
    if not os.path.isfile(fn):
        print('File not found! %s' %(fn))
        continue
    im = Image.open(fn)
    imwidth, imheight = im.size
    rindex = 0

    #===getting cordinates of patches shape 
    for region in via['_via_img_metadata'][fid]['regions']:
        if region['shape_attributes']['name'] != 'rect':
            print('extraction of %s regions not yet implemented!' % region['shape_attributes']['name'])
            continue
        x = region['shape_attributes']['x']
        y = region['shape_attributes']['y']
        w = region['shape_attributes']['width']
        h = region['shape_attributes']['height']

        left = max(0, x)
        top = max(0, y)
        right = min(imwidth, x + w)
        bottom = min(imheight, y + h)
        crop = im.crop((left, top, right, bottom))
        rindex = rindex + 1
        extold = os.path.splitext(via['_via_img_metadata'][fid]['filename'])[1]
        extnew = extold.replace('.', '_' + str(rindex) + '.')
        cropname = via['_via_img_metadata'][fid]['filename'].replace(extold, extnew)
        print(cropname)

        #===Convert image color space to different color spaces(RGB,HSV, LUV, HSL, LUV, YCrCb,YUV)
        crop = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2RGB)
        img = crop
        # rgb_float = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        img_rgb = img.reshape(-1, 3)
        hsv = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2HSV)
        img_hsv = hsv.reshape(-1, 3)
        # lab_float = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        lab = cv2.cvtColor(np.array(crop), cv2.COLOR_BGR2LAB)
        img_lab = lab.reshape(-1, 3)
        # lab_float = cv2.normalize(img, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        hsi = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2HLS)
        img_hsi = hsi.reshape(-1, 3)
        luv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2LUV)
        img_luv = luv.reshape(-1, 3)
        ybr = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2YCrCb)
        img_ybr = luv.reshape(-1, 3)
        yuv = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2YUV)
        img_yuv = yuv.reshape(-1, 3)

        R,G,B = cv2.split(img)
        r= R/(R+G+B)
        g= G/(R+G+B)
        b= B/(R+G+B)
        # final_nRGB= r+g+b
        hh = np.dstack((b, g, r))
        nx, ny, ns = hh.shape
        final_n = hh.reshape(nx * ny, ns)
        out_images = np.array(final_n.astype(int))
        final_nRGB = out_images.astype(np.uint8)

        #===Extracting individual pixels from the color spaces 
        data_attach = []
        for f in range(img_rgb.shape[0] and img_hsv.shape[0] and img_lab.shape[0] and img_hsi.shape[0] and img_luv.shape[0] and
                       img_ybr.shape[0] and img_yuv.shape[0] and final_nRGB.shape[0]):
            data = np.hstack([img_rgb[f, 0], img_rgb[f, 1], img_rgb[f, 2],
                              img_hsv[f, 0], img_hsv[f, 1], img_hsv[f, 2],
                              img_lab[f, 0], img_lab[f, 1], img_lab[f, 2],
                              img_hsi[f, 0], img_hsi[f, 1], img_hsi[f, 2],
                              img_luv[f, 0], img_luv[f, 1], img_luv[f, 2],
                              img_ybr[f, 0], img_ybr[f, 1], img_ybr[f, 2],
                              img_yuv[f, 0], img_yuv[f, 1], img_yuv[f, 2],
                              final_nRGB[f, 0], final_nRGB[f, 1], final_nRGB[f, 2]])

            labels = str(region['region_attributes'])
            labels2 = labels[12:16]

            #===Labeling the pixels as foreground or background 
            if labels2 == "fore":
                    data= np.append(data, int(1))
            elif labels2 == "back":
                    data= np.append(data, int(0))

            data_attach.append(data)
        final_data.extend((data_attach))

dd= pd.DataFrame(final_data)
datat2= dd

#===Storing data and label in hdf5 file
db = h5py.File('cowpea_200_only', 'w')
dataset = db.create_dataset('datasetName', data= datat2, dtype="float")
print(len(dataset))
db.close()



#===Reading the stored data n hdf5
filename = 'cowpea_200_only'
datasetss = h5py.File(filename, "r")
# (label5, data5) = (datasetss['datasetName'][:, 0], datasetss['datasetName'][:, 1:])
datass = (datasetss['datasetName'][:,:])


#===seperate data into labels and features
dat = datass[:,0:3]
labels = datass[:,-1]
data2= np.asarray(dat)
label2=np.asarray(labels)
# Splitting the dataset into the Training set and Test set




# data_link = 'D:\\Texting_texting000166.xlsx'

# dataset = pd.read_excel(data2)
#
# # ************************MACHINE LEARNING FOR TRAINING IMAGE PIXELS**************************************
# ===Data Preprocessing 
dataset.dropna(
    axis=0,
    how='any',
    thresh=None,
    subset=None,
    inplace=True)
# randomizing the rows for splitting purpose
data1 = dataset.sample(frac=1).reset_index(drop=True)

# Splitting the data into features and labels(features= dat, labels=labels)
dat = data1.iloc[:, 0: 1680].values
labels = data1.iloc[:, -1].values

#===Model performance comparison

# preparing  models(models include: LDA, RF, KNN,CART, NB, SVM)
models = []

models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('RF', RandomForestClassifier()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# evaluating  each model in turns
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=None)
    cv_results = model_selection.cross_val_score(model, data2, label2, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

# boxplot algorithm comparison
fig = plt.figure()
fig.suptitle('Algorithm Comparison')

ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)


plt.xlabel('Machine Learning Models')
plt.ylabel('Classification Accuracy')

#ax.xlabel('models')
#ax.ylabel('accuracy(%)')
plt.show()


#splitting the data into train and test data(70/30)
X_train, X_test, y_train, y_test = train_test_split(dat, labels, test_size=0.3, random_state=42)

# *********************TRAINING SVM MODEL **************************************************
svc = SVC()
params_grid = {'C': [0.1, 1, 10, 100], 'gamma': ['auto'], 'kernel': ['rbf', 'poly', 'sigmoid']}
cv = GridSearchCV(svc, params_grid, cv=5)
# fitting and estimating the best parameter in creating a model
cv.fit(X_train, y_train)
y_predict = cv.predict(X_test)
print(cv.best_estimator_)
# printing classification report
print(classification_report(y_test, y_predict))
# comparing predicted and actual results
print(pd.DataFrame(y_predict, y_test))
#==confusion matrix of the best estimator
plot_confusion_matrix(cv, X_test, y_test)
plt.show()
#==saving the SVM model
SVM_model = pickle.dumps(cv)
#SVM_from_pickle = pickle.loads(SVM_model)





#******************************TRAINING RANDOM FOREST MODEL***************************
classifier = RandomForestClassifier(n_estimators=300, random_state=0)
grid_param = {
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]}

model = GridSearchCV(estimator=classifier,
                     param_grid=grid_param,
                     scoring='accuracy',
                     cv=5,
                     n_jobs=-1)
model.fit(X_train, y_train)
# fitting and estimating the best parameter in creating a model
y_predict = model.predict(X_test)
best_parameters = model.best_params_
# printing best hyperparameter for training
print(best_parameters)
# printing classification report
print(classification_report(y_test, y_predict))
print(pd.DataFrame(y_predict, y_test))
# Drawing confusion matrix
plot_confusion_matrix(model, X_test, y_test)
plt.show()


#==Saving the RF model
RF_model = pickle.dumps(model)
RF_from_pickle = pickle.load(RF_model)