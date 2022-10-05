""" Prepare input data to feed to the segmentation model
Tharanghi Logendran, tharanghi@gmail.com"""

from matplotlib import pyplot as plt
import os
from os import listdir
import glob
import numpy as np
import cv2
import pandas as pd


images = listdir('C:/Netwerken/FromScratch/Image_segmentation/Frames/frames_ordered/')
masks = listdir('C:/Netwerken/FromScratch/Image_segmentation/Seg_masks/')
# make sure to have all frames ordered so that index in the folder matches the segmentation masks!

# prepare data as dataframe
#images.sort()
#masks.sort()
images_array = np.array([images])
images_array = images_array.T
masks_array = np.array([masks])
masks_array = masks_array.T
images_df = pd.DataFrame(images_array, columns = ['images'])
masks_df = pd.DataFrame(masks_array, columns = ['masks'])

matched = images_df.merge(masks_df, left_index=True, right_index=True)

# merge images and masks in 1 .png file to feed to the model
index=0
for i in matched.index:
    fig = plt.figure(constrained_layout=True, figsize=(10, 4))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.axes.xaxis.set_ticks([])
    ax1.axes.yaxis.set_ticks([])
    ax2.axes.xaxis.set_ticks([])
    ax2.axes.yaxis.set_ticks([])
    image_files = os.path.join('C:/Netwerken/FromScratch/Image_segmentation/Frames/frames_ordered/', str(images_df['images'][i]))
    image_files = cv2.imread(image_files)
    mask_files = os.path.join('C:/Netwerken/FromScratch/Image_segmentation/Seg_masks/', str(masks_df['masks'][i]))
    mask_files = cv2.imread(mask_files)
    ax1.imshow(image_files)
    ax2.imshow(mask_files)
    my_path = "D:/TGM3/Image_segmentation/Data/"
    fig.savefig(my_path + str(index))
    index += 1




