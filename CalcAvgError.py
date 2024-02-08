import cv2
import numpy as np
import os

def convert(img, target_type_min, target_type_max, target_type):
    imin = img.min()
    imax = img.max()

    a = (target_type_max - target_type_min) / (imax - imin)
    b = target_type_max - a * imax
    new_img = (a * img + b).astype(target_type)
    return new_img
def toGray(im, im_L, im_W, contrast=1.0):
    # normalize
    gray = np.zeros([im_L, im_W])
    for j in range(im_L):
        for i in range(im_W):
            gray[j][i] = max(im[j][i]) * contrast
    gray = convert(gray, 0, 255, np.uint8)
    return gray
def calcErrorAvg(dir):
    num_im = len(os.listdir(dir))
    if num_im == 0 or num_im == 1:
        return 0

    errorSum = 0
    for i in range(0, num_im):
        filename = dir + "/" + str(os.listdir(dir)[i])
        im0 = cv2.imread(dir + "/" + str(os.listdir(dir)[i]))
        im0 = toGray(im0, 10, 10)
        for j in range(0, num_im):
            im1 = cv2.imread(dir + "/" + str(os.listdir(dir)[j]))
            im1 = toGray(im1, 10, 10)
            errorSum += sum(sum(abs(im0.astype(int) - im1.astype(int))))

    errorAvg = errorSum / (num_im * num_im)
    return errorAvg


baseDir = 'ShapeGroups'
for filename in os.listdir(baseDir):
    print(filename + ": " + str(calcErrorAvg(baseDir + '/' + filename)))
