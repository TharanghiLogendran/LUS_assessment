""""Convert videos to frames for segmentation and make random selection"""""
""""Tharanghi Logendran, tharanghi@gmail.com"""

import cv2
import os, random, shutil

for file in os.listdir("D:/TGM3/Video_classification/Data"): # dir to all videos
    if file.endswith(".avi"):
        path=os.path.join("D:/TGM3/Video_classification/Data", file)
        vidcap = cv2.VideoCapture(path)
        success,image = vidcap.read()
        count = 0
        while success:
            name = "C:/Netwerken/FromScratch/Image_segmentation/Frames/all_frames/" + str(file) + str(count) + ".jpg"
            cv2.imwrite(name,image)  # save frame as JPEG file
            success, image = vidcap.read()
            # print('Read a new frame: ', success)
            count += 1

# make random selection
dirpath = 'C:/Netwerken/FromScratch/Image_segmentation/Frames/all_frames/' # dir containing all frames
destDirectory = 'C:/Netwerken/FromScratch/Image_segmentation/Frames/' # output dir

filenames = random.sample(os.listdir(dirpath), 800)
for fname in filenames:
    srcpath = os.path.join(dirpath, fname)
    shutil.copy(srcpath, destDirectory)



