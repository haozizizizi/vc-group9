import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

import random

# T1  start _______________________________________________________________________________
# Read in Dataset

# change the dataset path here according to your folder structure
dataset_path = "./images/"

X = []
y = []

for i in glob.glob(dataset_path + '*.png', recursive=True):
    # The string 'i' is split using "images" as the delimiter, resulting in a list.
    # The element at index 1 of the list is selected. eg.['/000_1_0001.png']
    # From the selected string, a substring is extracted from index 1 to index 3 (excluding index 3). eg.[000]
    # The extracted substring represents the label.
    label = i.split("images")[1][1:4]
    y.append(label)
    # write code to read ecah file i, and append it to list X
    image = cv2.imread(i)
    X.append(image)

# you should have X, y with 5998 entries on each.
# T1 end ____________________________________________________________________________________

# T2 start __________________________________________________________________________________
# Preprocessing
X_processed = []
for x in X:
    # Write code to resize image x to 48x48 and store in temp_x
    temp_x = cv2.resize(x, (48, 48))
    # Write code to convert temp_x  to grayscale 将图像转换为灰度图像
    gray_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
    # Append the converted image into X_processed 附加到X_processed列表
    X_processed.append(gray_x)

# T2 end ____________________________________________________________________________________

# T3 start __________________________________________________________________________________
# Feature extraction
X_features = []
for x in X_processed:
    x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                    cells_per_block=(1, 1), visualize=False, multichannel=False)
    X_features.append(x_feature)


# write code to Split training & testing sets using sklearn.model_selection.train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, shuffle=True)

# T3 end ____________________________________________________________________________________

# T4 start __________________________________________________________________________________
# train model
classifier = SVC()  # 初始化分类器
classifier.fit(x_train, y_train)  # 使用训练数据训练分类器

# 在测试集上评估分类器
accuracy = classifier.score(x_test, y_test)
print("准确率：", accuracy)