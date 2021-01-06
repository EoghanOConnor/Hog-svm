# Hog-svm
Hog svm sign dectector

Hog-svm has two files, trainer.py and detector.py

Trainer.py

Description: The trainer imports pickle files from the training dataset. 
             The dataset is divided into two subsets. Posex are images with a positive one 
             label. Negex are images with a negative one lebel. Both subsets change the 
             pixel dimensions of the images to 32x32, 40x40, 48x48, 64x64. The varied 
             images sizes are then converted into HOG descriptors using the HOG function. 
             These descriptors are then used to train four different LinearSVC. 
             These LinearSVC are then "dumped" into pickle files where they are used in 
             the slidingwindowdetector file.




Detector.py

Description: The sliding window detector program takes in an image as a 
             command line argument. THis image is scanned through and all detected 
             traffic signs are located. The program output three images to the local
             folder that highlights the locations of the detcted traffic signs. The 
             three images are also presented to the user. Information about the bounding
             boxes for eact detected traffic sign is printed to the console.
