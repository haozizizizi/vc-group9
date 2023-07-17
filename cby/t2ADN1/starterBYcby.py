import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "D:\\Dataset_1\\images\\"

X = []
y = []
for i in glob.glob(dataset_path + '*.png', recursive=True):
    label = i.split("images")[1][1:4]
    y.append(label)

    # write code to read ecah file i, and append it to list X
'''by cby'''
for i in glob.glob(dataset_path + '*.png',recursive=False):
    images = cv2.imread(dataset_path)
    X.append(images)
# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________


# T2 start __________________________________________________________________________________
# Preprocessing
X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Write code to convert temp_x to grayscale

    # Append the converted image into X_processed


# T2 end ____________________________________________________________________________________


# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, multichannel=False)
    X_features.append(x_feature)

# write code to Split training & testing sets using sklearn.model_selection.train_test_split


#T3 end ____________________________________________________________________________________



#T4 start __________________________________________________________________________________
# train model
