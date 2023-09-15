from __future__ import absolute_import, division, print_function

from scipy import ndimage, signal
import cv2
import os

import numpy as np

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
            # if max(im[j][i]) > 0:
            #     gray[j][i] = 255
            # else:
            #     gray[j][i] = 0
    # gray = convert(gray, 0, 255, np.uint8)
    return gray

def threshold(frame):
    # for j in range(len(frame)):
    #     for i in range(len(frame[0])):
    #         print(frame[j][i])
    #
    # print("---------------------------------")
    out = np.full_like(frame, 128)
    out[frame < -4] = 0
    out[frame > 4] = 255
    return out

def SpacialContrast(imGray):
    conv_gray = signal.convolve2d(imGray, kSC, 'same')
    fSCgray = np.array(threshold(conv_gray), dtype=np.uint8)
    return fSCgray


if os.path.isdir("ShapeGroups"):
    os.system("rm -rf ShapeGroups")
else:
    os.makedirs("ShapeGroups")

# go thru each available folder and calc error (abs(im2-im1))
# if none make one!
# pick one with lowest error and return that directory as shape_folder
# hopefully if all goes well, we will have 6 folders with properly binned shapes
def putShapeInFolder(shape, shape_id, baseDir = "ShapeGroups",
                     contrast=1, max_acceptable_error=1E3, pixel_ratio=1.0):
    H_shape = 10
    W_shape = 10
    num_folders = len(os.listdir("ShapeGroups"))
    # print(shape_id)

    # if the ShapeGroups Folder is empty
    if num_folders == 0:
        shape_folder = baseDir + "/" + "shape" + str(num_folders) + "/"
        if not os.path.isdir(shape_folder):
            os.makedirs(shape_folder)
        cv2.imwrite(shape_folder + shape_id + ".png", shape)
        return  # get out!!

    # get the avg error from each folder
    errorCategorical = np.empty(shape=num_folders, dtype=float)
    for idx in range(num_folders):
        current_folder = os.listdir("ShapeGroups")[idx]
        num_ims = len(os.listdir(baseDir + "/" + current_folder))
        # print("num ims " + str(num_ims))
        errors = np.zeros(shape=num_ims, dtype=float)
        for jdx in range(num_ims):
            ims_list = os.listdir(baseDir + "/" + current_folder)
            filename = ims_list[jdx]
            # print(baseDir + "/" + current_folder + "/" + filename)
            shapeCurr = cv2.imread(baseDir + "/" + current_folder + "/" + filename)
            shapeCurr = toGray(shapeCurr, im_L=10, im_W=10, contrast=contrast)
            # print(shapeCurr)
            errors[jdx] = sum(sum(abs(shape.astype(int)/255 - shapeCurr.astype(int)/255)))
            # print(filename + " error: " + str(abs(shape.astype(int)/255 - shapeCurr.astype(int)/255)))

        #average the errors
        errorCategorical[idx] = np.average(errors)
        # print(os.listdir("ShapeGroups")[idx] + " error: " + str(errorCategorical[idx]))
    # print("min @ " + str(np.where(errorCategorical == errorCategorical.min())[0][0]))

    # if the min error is above the maximum acceptable error, make a new shape
    if min(errorCategorical) < max_acceptable_error:
        found_folder = os.listdir("ShapeGroups")[(np.where(errorCategorical == errorCategorical.min())[0][0])] + "/"
        filename = baseDir + "/" + found_folder + shape_id + ".png"
        cv2.imwrite(filename, shape)
        # print("FOUND: " + filename + " error: " + str(min(errorCategorical)))
    else:
        new_folder = baseDir + "/" + "shape" + str(num_folders) + "/"
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder)
        filename = new_folder + shape_id + ".png"
        cv2.imwrite(filename, shape)
        # print("NEW:" + filename + " error: " + str(min(errors)))
    return  # get out!!


# get dataset!
image_sets = ['train.large', 'train.med', 'train.small', 'train.tiny']
training_image_files = './exp_shapes/shapes_dataset/%s.input.npy'
training_images_list = []

for image_set in image_sets:
    training_images_list.append(np.load(training_image_files % image_set))

training_images = np.concatenate(training_images_list)

# joe code starts here
# --------------------------------------------------------------
kSC = np.array([[-1 / 8, -1 / 8, -1 / 8],  # from class material
                [-1 / 8, 1, -1 / 8],
                [-1 / 8, -1 / 8, -1 / 8]])

folder = "Sample List/"
contrast = 1.3
for sample in range(121, 180):
    print("Sample " + str(sample))
    gray = toGray(im=training_images[sample], im_L=30, im_W=30, contrast=contrast)

    shape_folder = folder + "sample" + str(sample) + "/"
    if not os.path.isdir(shape_folder):
        os.makedirs(shape_folder)

    # loops over each shape in 3x3 grid
    for j in range(3):
        for i in range(3):
            # make shape_id
            shape_id = str(sample) + "_" + str(j) + str(i)

            # grayscale & SC the image
            shape = gray[(j * 10):(j * 10 + 10), (i * 10):(i * 10 + 10)]
            shapeSC = SpacialContrast(shape)

            # Save the SC image with the rest of that sample and then find a group for it
            cv2.imwrite(shape_folder + "fSC" + shape_id + ".png", shapeSC)

            putShapeInFolder(shapeSC, shape_id, max_acceptable_error=16, pixel_ratio=255)


