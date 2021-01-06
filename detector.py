#!/usr/bin/env python
# coding: utf-8

# 
# 
# Authors :     Eoghan O'Connor    Luke Vickery
# Student I.Ds: 16110625           16110501
# 
# File Name: detector.py ,
# 
# Description:
# 
# Inputs: Image file name
#
# Description: The sliding window detector program takes in an image as a 
# command line argument. THis image is scanned through and all detected 
# traffic signs are located. The program output three images to the local
# folder that highlights the locations of the detcted traffic signs. The 
# three images are also presented to the user. Information about the bounding
# boxes for eact detected traffic sign is printed to the console.
#
# Output: Saves three .png files to local folder
#         Presents three images to user

import sys
import numpy as np
from numpy import dstack
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.feature import hog
import pickle
from PIL import Image       
from os import path


# # Function that takes in an image and a window size. returns a numpy array of
# all windows of defined size and an array of matching coordinates.
def window_kernel(Image,NxNsize):
    ppc=NxNsize/8
    h,w,c= Image.shape
    windows=[]
    coords = []
    for row in range(0,(h-NxNsize),4):
        for col in range(0,(w-NxNsize),4):
            window=np.array(Image[row:row+NxNsize,col:col+NxNsize],dtype='float32')
            
            hog_i = hog(window, orientations=9,   # 9 orientation bins
                      pixels_per_cell=(ppc,ppc),  # variable pixel subregions
                       cells_per_block=(2,2),     # 2*2 subregion merge
                       visualize=False,           # show visualization
                       multichannel=True)         # input is RGB
            
            coords.append([col,row])
            windows.append(hog_i)
    return windows, coords

# # Function to craete bounding box square that van be overlaid onto an image
# Takes in a window size and the coordinates of the top left corner
def create_square(NxNsize,col,row):
        colour_list={64:'r',48:'g',40:'m',32:'y'}
        colour= colour_list.get(NxNsize)
        square = mpl.patches.Rectangle((col,row),
                                              NxNsize,NxNsize, 
                                              edgecolor=colour, 
                                              facecolor="none")
        return square

# # This function calculate the ratio fo the intersection over union.
# Two boxes are passed in as arrays containing the x coordinate and y 
# coordinate of the top left corner as the first and second element of the 
# arrays respectively. The ratio of overlap is return from the function
def intersection_over_union(box1, box2):
    # determine the (x, y)-coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[0]+box1[3], box2[0]+box2[3])
    y2 = min(box1[1]+box1[3], box2[1]+box2[3])

    # compute the area of intersection rectangle
    intersectArea = abs(max((x2 - x1, 0)) * max((y2 - y1), 0))
    if intersectArea == 0:
        return 0
    # compute the area of both the first and second
    # rectangles
    box1Area = abs((box1[0]+box1[3] - box1[0]) * (box1[1]+box1[3] - box1[1]))
    box2Area = abs((box2[0]+box2[3] - box2[0]) * (box2[1]+box2[3] - box2[1]))
    unionArea = float(box1Area + box2Area - intersectArea)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of first and second box areas areas - 
    # the interesection area
    ratio = intersectArea / unionArea

    # return the intersection over union value
    return ratio

# # This function takes in two arrays, one containing all positive activation
# as the first argument and the seconf is an array to store local maximum
# activation windows in. It uses the intersection over union function to 
# perform non maximum suppresion in clusters of activation windows around
# a detected sign
def rec_function(array, max_arr):
    if len(array) == 0:
        return
    else:
        temp_arr = []
        temp_leftover = []
        temp_arr.append(array[0])
        # Iterate over all elements. Compare to all others
        for i in range(1,len(array)):  
            if intersection_over_union(array[0],array[i]) > 0:
                temp_arr.append(array[i])
            else:
                temp_leftover.append(array[i])

        cluster_max = np.sort(temp_arr, order = 'activ')
        max_arr.append(cluster_max[-1])
        if len(temp_leftover) > 0:
            rec_function(temp_leftover, max_arr)
        else:
            return


# # Import image from console argument
if len(sys.argv) != 2:
    print('WARNING!! - run code using following format: python "program name" "image file name" .jpg')
    print('It is expected that images are contained in the folder "hog-svm-testimages/"')
    exit()

# Generate image file path 
image_path = str("hog-svm-testimages/" + sys.argv[1])

# # Error check to ensure file path exists
if path.exists("hog-svm-testimages/"+ sys.argv[1]):
    im1 = np.array(Image.open(image_path)) # Open the image and convert to a Numpy array.
else:
    print("WARNING!! - File path not found. Ensure images are at following file path: hog-svm-testimages/image.jpg")
    exit()

# # Loading of all four classifiers trained in the training program
classifier64 = pickle.load(open("linsvmhog64.pkl","rb"))
classifier48 = pickle.load(open("linsvmhog48.pkl","rb"))
classifier40 = pickle.load(open("linsvmhog40.pkl","rb"))
classifier32 = pickle.load(open("linsvmhog32.pkl","rb"))


# # Generating Activation Level Arrays
# For each size of sliding window the activation levels for each and every 
# sliding window is calculated. This is a slow process and must be run for each 
# new image. This process could be made much faster by implementing a larger
# step size resulting in less windows to calculate histogram of gradient 
# decents for.

print("Compiling array of all 64x64 sliding windows")
window_hog64, window_crd64 = window_kernel(im1,64)
activation_array64 = classifier64.decision_function(window_hog64)

print("Compiling array of all 48x48 sliding windows")
window_hog48, window_crd48 = window_kernel(im1,48)
activation_array48 = classifier48.decision_function(window_hog48)

print("Compiling array of all 40x40 sliding windows")
window_hog40, window_crd40 = window_kernel(im1,40)
activation_array40 = classifier40.decision_function(window_hog40)

print("Compiling array of all 32x32 sliding windows")
window_hog32, window_crd32 = window_kernel(im1,32)
activation_array32 = classifier32.decision_function(window_hog32)


# # Find Max Activation Level
# The maximum activation level is found across all four sliding window sizes. 
# The size of the sliding window from which this was found is not yet relevant so it is not stored as a variable.
# The four sliding window activation level arrays are stacked together
# The max value is returned and cast to a float value for printing purposes
# The cutoff threshold value is defined
# These values are printed to the console

print("Culling all activation levels below 0.2*Max Activation")
stacked_act = dstack((np.max(activation_array64),
                      np.max(activation_array48),
                      np.max(activation_array40),
                      np.max(activation_array32)))
max_activation = float(stacked_act.max(2))
cutoff_threshold = 0.2*max_activation
print(f"Max activation: {max_activation}, the cutoff thresehold is defined as: {cutoff_threshold}")


# # Create Lists of Activations & Matching Coords
# Lists are generated for all activations greater than the defined threshold 
# activation level along with their matching coordinates. These lists are 
# created for each size of sliding window sizes.

# # Initialise arrays to store valid activation windows
val_arr_all = []
val_arr48 = []
val_arr64 = []
val_arr40 = []
val_arr32 = []

# # Loop through all activation windows and perform threshold culling
# This process is repeated for each window size

for level in activation_array64:
    if level >= cutoff_threshold:
        element = np.where(activation_array64==level)
        data = ((window_crd64[int(element[0])][0], 
                window_crd64[int(element[0])][1], 
                level, 
                64))
        val_arr64.append(data)
        val_arr_all.append(data)

for level in activation_array48:
    if level >= cutoff_threshold:
        element = np.where(activation_array48==level)
        data = ((window_crd48[int(element[0])][0], 
                window_crd48[int(element[0])][1], 
                level, 
                48))
        val_arr48.append(data)
        val_arr_all.append(data)     

for level in activation_array40:
    if level >= cutoff_threshold:
        element = np.where(activation_array40==level)
        data = ((window_crd40[int(element[0])][0], 
                window_crd40[int(element[0])][1], 
                level, 
                40))
        val_arr40.append(data)
        val_arr_all.append(data)    

for level in activation_array32:
    if level >= cutoff_threshold:
        element = np.where(activation_array32==level)
        data = ((window_crd32[int(element[0])][0], 
                window_crd32[int(element[0])][1], 
                level, 
                32))
        val_arr32.append(data)
        val_arr_all.append(data)


# # Generate Image Overlay
# The valid activation level coordinate values are used to generate the drawn 
# boxes to overlay on the image. In this image all valid activation 
# levels for all window sizes are drawn

figure, ax = plt.subplots(1)
for val in val_arr_all:
    box = create_square(val[3],val[0],val[1])
    ax.add_patch(box)
ax.imshow(im1)
ax.title.set_text('All Activation Levels > 0.2*Max')
plt.savefig(sys.argv[1][0:-4] + "_all_activations.png")

# # Create datatype arrays of activation levels
# A defined dataype is applied to each array of valid activation levels

dtype = [('x', int), ('y', int), ('activ', float), ('size', int)]

# Data arrays are created with defined datatypes
data64 = np.array(val_arr64, dtype=dtype)
data48 = np.array(val_arr48, dtype=dtype)
data40 = np.array(val_arr40, dtype=dtype)
data32 = np.array(val_arr32, dtype=dtype)
dataAll = np.array(val_arr_all, dtype=dtype)

# # Create arrays of max activations per sign
# The function rec_function is used to cull all overlapping activation windows
# That are not a local maximum. This is completed for each of the four window 
# sizes to enable plotting of max activation per sign for each window size.
# data_all is used to perform non maximum suppression of all but the strongest
# activation per sign regardless of window size

max_per_size = []
max_overall = []

rec_function(data64, max_per_size)
rec_function(data48, max_per_size)
rec_function(data40, max_per_size)
rec_function(data32, max_per_size)
rec_function(dataAll, max_overall)

# # Plotting of max activations per window size for each sign
# Info of max activations for each window size is also printed to console
print("The max activation window per window size is as follows")

figure2, ax2 = plt.subplots(1)
for val in max_per_size:
    box = create_square(val[3],val[0],val[1])
    ax2.add_patch(box)
    print(f"Sign detected: Bounding box defined at: ({val[0]},{val[1]}   (x,y))")
    print(f"               Height/Width:            {val[3]}x{val[3]}    (pixels)")
    print(f"               Activation Level:        {val[2]}\n")
ax2.imshow(im1)
ax2.title.set_text('Highest Activation Per Sign & Window Size')
plt.savefig(sys.argv[1][0:-4] + "_max_per_size.png")


# # Plotting of max activations per sign
# Info of max activations is also printed to console
figure3, ax3 = pltfigure2, ax2 = plt.subplots(1)
print("The bounding boxes of best fit are:\n")
for val in max_overall:
    box = create_square(val[3],val[0],val[1])
    ax3.add_patch(box)
    print(f"Sign detected: Bounding box defined at: ({val[0]},{val[1]}   (x,y))")
    print(f"               Height/Width:            {val[3]}x{val[3]}    (pixels)")
    print(f"               Activation Level:        {val[2]}\n")
ax3.imshow(im1)
ax3.title.set_text('Max Activation Box Per Sign')
plt.savefig(sys.argv[1][0:-4] + "_max_overall.png")
print("Figures 1-3 saved as .png files in local folder")

# # Display each of the three image to the user
plt.show()




