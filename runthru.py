import cv2
import os
import time
import numpy as np
# %%
training_set = np.empty([120*9, 10, 10, 3])
src_set = np.empty(shape=120*9, dtype=object)
baseDir = 'ShapeGroups'
idx = 0
for filename in os.listdir(baseDir):
    folder = baseDir + '/' + filename
    for imagepath in os.listdir(folder):
        imagepathFull = folder + '/' + imagepath
        print(imagepathFull)
        training_set[idx] = cv2.imread(imagepathFull)
        src_set[idx] = imagepathFull
        idx += 1
# %%
cv2.imshow("", cv2.resize(training_set[0], [30, 30]))
cv2.waitKey(0)

idx = 0
for sample in training_set:
    cv2.imshow("", cv2.resize(sample, [30, 30]))
    print(src_set[idx])
    cv2.waitKey(0)
    idx += 1
# %%
