import cv2
import numpy as np


#图像滤波

def pre_Blur(img,ksize):
    '''
    均值滤波 -- 取内核区域下所有像素的平均值并替换中心元素
        特点：内核中区域贡献率、权重相同
        作用：对于椒盐噪声滤除效果好
    :param img:
    :param ksize: 核大小
    :return:
    '''
    blur = cv2.blur(img,(ksize,ksize))
    return blur

def pre_boxBlur(img,ksize,normalize:bool):
    '''
    方框滤波 --
        当 normalize=true：与均值滤波相同
        当 normalize=false：表示对加和后结果不进行平均（像素值最大为255）
    :param img:
    :param ksize:
    :param normalize:
    :return:
    '''
    boxBlur = cv2.boxFilter(img,-1,(ksize,ksize),normalize)
    return boxBlur


def pre_gaussianBlur(img,ksize,sigmaX:int,sigmaY:int):
    '''
    高斯滤波

    :param img:
    :param ksize:
    :param sigmaX: X方向的标准偏差
    :param sigmaY: Y方向上的标准偏差
    :return:
    '''
    gaussian_blur = cv2.GaussianBlur(img,(ksize,ksize),sigmaX,sigmaY)
    return gaussian_blur

def pre_medianBlur(img,ksize):
    '''
    中值滤波 -- 内核内的像素值进行排序，取中值作为当前值
        特点：中心点的像素被内核中位数替代
        作用：对于椒盐噪声邮箱
    :param img:
    :param ksize:
    :return:
    '''

    medianBlur = cv2.medianBlur(img,ksize)
    return medianBlur


# 直方图均衡化
def pre_singleChannel_equalHist():
    '''
    单通道的直方图均衡化
    :param img:
    :return:
    '''
    img = cv2.imread('lena.png',0)

    singleChannel_eq = cv2.equalizeHist(img)
    # singleChannel_equalHist_img = np.hstack((img,singleChannel_eq))
    imgshow('eq',singleChannel_eq)
    # imgshow('img',singleChannel_equalHist_img)
    # return singleChannel_equalHis
#2值化
def pre_Binary(img):

    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dst = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    imgshow('Binary',dst)
    return dst

# 霍夫直线检测
def pre_HoughLinesP(binary_img):
    #输入图像首先必须是二值图像

def imgopen(path):
    img = cv2.imread(path)
    return img
def imgshow(name:str,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    img = imgopen('lena.png')
    imgshow('lena', img)
    Blur = pre_Blur(img,5)
    imgshow('Blur',Blur)
    gaussianBlur = pre_gaussianBlur(img,11,0,0)
    imgshow('gaussianBlur',gaussianBlur)
    medianBlur = pre_medianBlur(img,5)
    imgshow('medianBlur',medianBlur)
    pre_singleChannel_equalHist()
    binary_img = pre_Binary(img)


