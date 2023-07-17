import cv2
import sklearn
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import random

dataset_path = "Dataset_1\\images\\"
X = []
y = []
X_processed = []
X_features = []


def loaddata(dataset_path):
    for i in glob.glob(dataset_path + '*.png', recursive=True):
        label = i.split("images")[1][1:4]
        y.append(label)
    print(y[5777])

    # write code to read ecah file i, and append it to list X
    '''by cby'''
    for i in glob.glob(dataset_path + '*.png', recursive=False):
        images = cv2.imread(i)
        X.append(images)
    # print(X)
    print('loaddata success!')


def turn2gray():
    for x in X:
        '''resize 48*48'''
        temp_x = cv2.resize(x, (48, 48))
        '''by CBY'''  # Write code to convert temp_x to grayscale
        gray_x = cv2.cvtColor(temp_x, cv2.COLOR_BGR2GRAY)
        '''by CBY'''  # Append the coverted image into X_processed
        X_processed.append(gray_x)


def FeatureExtraction():
    # FeatureExtraction
    for x in X_processed:
        x_feature = hog(x, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(1, 1), visualize=False)
        X_features.append(x_feature)

    # write code to Split training & testing sets using sklearn.model_selection.train_test_split
    '''by CBY'''
    X_train, X_test, y_train, y_test = train_test_split(X_features,y, test_size=0.2, shuffle=True)
    return X_train,X_test,y_train,y_test




def random_shuffle(data, label):
    randnum = np.random.randint(0, 1234)
    np.random.seed(randnum)
    np.random.shuffle(data)
    np.random.seed(randnum)
    np.random.shuffle(label)
    return data, label


def random_show_shuffle(data,label):
    randnum = random.randint(0,1000)
    print(label[randnum])
    print(data[randnum])
    # cv2.imread(data)
    cv2.imshow('random',data[randnum])

    # cv2.imshow(label[randnum],data[randnum])
    cv2.waitKey(0)
    cv2.destroyWindow('random')


def tryshowimag():
    RDM = random.randint(200, 4000)
    print(RDM)
    cv2.imshow('RDMimg', X_processed[RDM])
    cv2.waitKey(0)
    cv2.destroyWindow('RDMimg')

def Trainmode(X_train,y_train,X_test,y_test):
    classfier = SVC()
    classfier.fit(X_train,y_train)

    # 在测试集上评估分类器
    accuracy = classfier.score(X_test,y_test)
    print(accuracy)

def main():
    loaddata(dataset_path)
    turn2gray()
    tryshowimag()
    X_train,X_test,y_train,y_test = FeatureExtraction()
    Trainmode(X_train = X_train, X_test = X_test,y_train = y_train, y_test = y_test)
    # random_show_shuffle(Shuffle_X_train,Shuffle_y_train)


if __name__ == '__main__':
    main()
