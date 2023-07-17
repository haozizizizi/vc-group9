import cv2
import sklearn
from skimage import io
import glob
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
import random
from matplotlib import image



def loaddata(X,y,dataset_path):
    '''

    :param X: data
    :param y: label
    :param path:
    :return: 数据集X 标签集y
    '''

    for i in glob.glob(dataset_path + '*.png', recursive=True):
        label = i.split("images")[1][1:4]
        y.append(label)
    # print(y[5777])

    # write code to read ecah file i, and append it to list X
    '''by cby'''
    for i in glob.glob(dataset_path + '*.png', recursive=False):
        images = cv2.imread(i)
        X.append(images)
    # print(X)
    print('loaddata success!')

    return X,y

def split_tainANDtest(X,y,X_train,X_test,y_train,y_test,test_rate,shuffle:bool):
    '''
    将总集合分类成测试集合和数据集合
    :param X:
    :param y:
    :param X_train:
    :param X_test:
    :param y_train:
    :param y_test:
    :param test_rate:
    :param shuffle:
    :return:
    '''
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_rate,shuffle=shuffle)
    return X_train, X_test, y_train, y_test

def showimgANDlabel(X,y,num:int):
    '''

    :param num: 下标
    :return: None
    '''
    img = image.imread(X)
    img.open()
    # image.i

