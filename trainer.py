#!/usr/bin/env python
# coding: utf-8

# 
# 
# Authors :     Eoghan O'Connor    Luke Vickery
# Student I.Ds: 16110625           16110501
# 
# File Name: trainer.py ,
# 
# Inputs: pickle file of training images
# 
# Description: The trainer imports pickle files from the training dataset. 
# The dataset is divided into two subsets. Posex are images with a positive one 
# label. Negex are images with a negative one lebel. Both subsets change the 
# pixel dimensions of the images to 32x32, 40x40, 48x48, 64x64. The varied 
# images sizes are then converted into HOG descriptors using the HOG function. 
# These descriptors are then used to train four different LinearSVC. 
# These LinearSVC are then "dumped" into pickle files where they are used in 
# the slidingwindowdetector file.
#
# Outputs: Classifers for each window size saved as pickle files


# Import necessary libraries

import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from scipy import ndimage
import pickle


# # Function to return the HoG for a supplied training image and resize if
# necessary. AN image and a size are passed to the function and a HoG is 
# returned.
def hog_trainer(image, size):
    if size == 64:
        # default settings for passed in image
        ppc = 8
        im1 = image
    else:
        # define variable pixel subregion size based on passed in size variable
        ppc = size//8
        # Resize all windows when size variable not equl to 64
        im1=ndimage.zoom(image,[size/64,size/64,1])
    
    # Return the histogram of gradients for the image passed in
    return hog(im1, orientations=9,              # 9 orientation bins
                      pixels_per_cell=(ppc,ppc), # pixel subregions
                       cells_per_block=(2,2),    # 2*2 subregion merge
                       visualize=False,          # show visualization
                       multichannel=True)        # input is RGB

# # Loading of dataset from .npz file
# The pickle file of training images is loaded and the psitive and negative
# training images ectracted.
dataset = np.load("hog-svm-train-dataset.64x64x3.npz", allow_pickle=True)  # Load the npz file.
posex = dataset['posex']
negex = dataset['negex']

# # All training images and matching labels grouped together. Shape of training set is printed
all_ex = np.concatenate((posex, negex))
all_labels = np.concatenate((np.ones(len(posex)),-np.ones(len(negex))))   
print("The shape of the training images dataset is:")
print(all_ex.shape)

# # 64x64 HoG for each training image is created and stored in an array
print("\nTraining 64x64 classifier")
all_hogs64 = []
for ex in all_ex:
    all_hogs64.append(hog_trainer(ex, 64))

# HoG array converted to numpy array
all_hogs64=np.array(all_hogs64)

# Classifier initialised and trained on training set HoG, model saved as pickle file
classifier64= LinearSVC()
classifier64.fit(all_hogs64, all_labels)
pickle.dump(classifier64, open("linsvmhog64.pkl","wb"))

# # 48x48 HoG for each training image is created and stored in an array
print("Training 48x48 classifier")
all_hogs48 = []
for ex in all_ex:
    all_hogs48.append(hog_trainer(ex, 48))

# HoG array converted to numpy array
all_hogs48=np.array(all_hogs48)

# Classifier initialised and trained on training set HoG, model saved as pickle file
classifier48= LinearSVC()
classifier48.fit(all_hogs48, all_labels)
pickle.dump(classifier48, open("linsvmhog48.pkl","wb"))

# # 40x40 HoG for each training image is created and stored in an array
print("Training 40x40 classifier")
all_hogs40 = []
for ex in all_ex:
    all_hogs40.append(hog_trainer(ex, 40))

# HoG array converted to numpy array
all_hogs40=np.array(all_hogs40)

# Classifier initialised and trained on training set HoG, model saved as pickle file
classifier40= LinearSVC()
classifier40.fit(all_hogs40, all_labels)
pickle.dump(classifier40, open("linsvmhog40.pkl","wb"))

# # 32x32 HoG for each training image is created and stored in an array
print("Training 32x32 classifier")
all_hogs32 = []
for ex in all_ex:
    all_hogs32.append(hog_trainer(ex, 32))

# HoG array converted to numpy array
all_hogs32=np.array(all_hogs32)

# Classifier initialised and trained on training set HoG, model saved as pickle file
classifier32= LinearSVC()
classifier32.fit(all_hogs32, all_labels)
pickle.dump(classifier32, open("linsvmhog32.pkl","wb"))


print("Training Complete")