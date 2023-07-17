

import cv2
import random

def randomshow_gray(X):
    rdm = random.randint(0, 5998)
    print('rdm = ', rdm)
    print(X[rdm].shape)
    cv2.imshow('rdmshow', X[rdm])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def radnomshow_3channal(X):
    rdm = random.randint(0, 5998)
    print('rdm = ', rdm)
    print(X[rdm].shape)
    cv2.imshow('rdmshow', X[rdm])
    cv2.imshow('rdmshow_B', X[rdm][:, :, 0])
    cv2.imshow('rdmshow_G', X[rdm][:, :, 1])
    cv2.imshow('rdmshow_R', X[rdm][:, :, 2])
    cv2.waitKey(0)
    cv2.destroyAllWindows()