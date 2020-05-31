import cv2
import numpy as np


def cut_image(img, mask):
    imshape = img.shape()
    # print(imshape) = (512, 512)
    for i in range(0, imshape[0]):  # imshape[0] = 512
        for j in range(0, imshape[1]):  # imshape[1] = 512
            if mask.data[i][j] <= 20:
                img.data[i][j] = 0
    return img


"""
args[0] = image
args[1] = mask
"""


def job(args):
    print("New job started")
    #img = cv2.equalizeHist(img)
    img = cut_image(args[0], args[1])
    print("Job finished")
    return img
