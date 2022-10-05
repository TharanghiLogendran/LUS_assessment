# example of pix2pix gan for satellite to map image-to-image translation
from numpy import load
from numpy import zeros
from numpy import ones
from numpy.random import randint
from numpy import load
from numpy import vstack
from numpy import expand_dims
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import LeakyReLU
from keras.models import load_model
from keras import metrics
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import vstack
from numpy import savez_compressed
from numpy import load
import tensorflow as tf
import numpy as np
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import mean_squared_error
import cv2
from skimage import color
from skimage import io
import os, random, shutil
from os import listdir
from PIL import Image

# load, split and scale the dataset ready for training
# load all images in a directory into memory
def load_images(path, size=(256,512)):
	src_list, tar_list = list(), list()
	# enumerate filenames in directory, assume all are images
	for filename in listdir(path):
		# load and resize the image
		pixels = load_img(path + filename, target_size=size)
		# convert to numpy array
		pixels = img_to_array(pixels)
		# split into satellite and map
		sat_img, map_img = pixels[:, :256], pixels[:, 256:]
		src_list.append(sat_img)
		tar_list.append(map_img)
	return [asarray(src_list), asarray(tar_list)]

# example of loading a pix2pix model and using it for image to image translation

# load and prepare training images
def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

# plot source, generated and target images
def plot_images(src_img, gen_img, tar_img):
	images = vstack((src_img, gen_img, tar_img))
	# scale from [-1,1] to [0,1]
	images = (images + 1) / 2.0
	titles = ['Source', 'Generated', 'Expected']
	# plot images row by row
	for i in range(len(images)):
		# define subplot
		plt.subplot(1, 3, 1 + i)
		# turn off axis
		plt.axis('off')
		# plot raw pixel data
		plt.imshow(images[i])
		# show title
		plt.title(titles[i])
	plt.show()

# dataset path
path = 'E:/M3-LUMC/Segmentatiedata/Data_segmentation/output/Val/'
# load dataset
[src_images, tar_images] = load_images(path)
print('Loaded: ', src_images.shape, tar_images.shape)
# save as compressed numpy array
filename = 'testlus.npz'
savez_compressed(filename, src_images, tar_images)
print('Saved dataset: ', filename)

# load dataset
[X1, X2] = load_real_samples('C:/Netwerken/FromScratch/testlus.npz')
print('Loaded', X1.shape, X2.shape)
# load model
model = load_model("E:/M3-LUMC/Segmentatieresultaten/model_000900.h5")
# select random example
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
# generate image from source
gen_image = model.predict(src_image)
# plot all three images
plot_images(src_image, gen_image, tar_image)
#%% example of loading a pix2pix model and using it for one-off image translation

# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels


# load source image
src_image = load_image('E:/M3-LUMC/Segmentatiedata/Data_segmentation/output/Val/200.png')
print('Loaded', src_image.shape)
# load model
model = load_model('C:/Netwerken/FromScratch/Image_segmentation/SavedModels/model_000900.h5')
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
# plot the image
plt.imshow(gen_image[0])
plt.axis('off')
plt.show()

#%% Metrics
src_images, tar_images = X1, X2
gen_images = model.predict(src_images)
tar_images = color.rgb2gray(tar_images)
gen_images = color.rgb2gray(gen_images)

#%%
# normalize pixel values to [0,1]
for nr,img in enumerate(tar_images):
    min = np.min(img)
    max = np.max(img)
    tar_images[nr] = (img - min) / (max - min)

for nr,img in enumerate(gen_images):
    min = np.min(img)
    max = np.max(img)
    gen_images[nr] = (img - min) / (max - min)

# Mean squared error
for i in range(len(gen_images)):
	mse = mean_squared_error(gen_images[i], tar_images[i])
	print("Mean squared error: {}".format(mse))

# Dice similarity function
# https://www.codegrepper.com/code-examples/python/code+to+calculate+dice+score
def single_dice_coef(y_true, y_pred_bin):
	# shape of y_true and y_pred_bin: (height, width)
	intersection = np.sum(y_true * y_pred_bin)
	if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
		return 1
	return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

for i in range(len(gen_images)):
	dice_score = single_dice_coef(tar_images[i], gen_images[i])
	print ("Dice Similarity: {}".format(dice_score))

#%% corrected for black spaces on the sides

# normalize pixel values to [0,1]
for nr,img in enumerate(tar_images):
	min = np.min(img)
    max = np.max(img)
    tar_images[nr] = (img - min) / (max - min)

for nr,img in enumerate(gen_images):
    min = np.min(img)
    max = np.max(img)
    gen_images[nr] = (img - min) / (max - min)

# cut out box
def crop_center(img,cropx,cropy):
    y,x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx]

gen_cropped = []
tar_cropped = []
for i in range(len(gen_images)):
	gen_cropped.append(crop_center(gen_images[i],96,256))
	tar_cropped.append(crop_center(tar_images[i],96,256))

gen_cropped = np.array(gen_cropped)
tar_cropped = np.array(tar_cropped)
gen_cropped = gen_cropped.round(decimals=0, out=None)
tar_cropped = tar_cropped.round(decimals=0, out=None)

# Mean squared error
for i in range(len(gen_cropped)):
	mse = mean_squared_error(gen_cropped[i], tar_cropped[i])
	print("Mean squared error: {}".format(mse))

# Dice similarity function
# https://www.codegrepper.com/code-examples/python/code+to+calculate+dice+score
def single_dice_coef(y_true, y_pred_bin):
	# shape of y_true and y_pred_bin: (height, width)
	intersection = np.sum(y_true * y_pred_bin)
	if (np.sum(y_true) == 0) and (np.sum(y_pred_bin) == 0):
		return 1
	return (2 * intersection) / (np.sum(y_true) + np.sum(y_pred_bin))

for i in range(len(gen_cropped)):
	dice_score = single_dice_coef(tar_cropped[i], gen_cropped[i])
	print ("Dice Similarity: {}".format(dice_score))
