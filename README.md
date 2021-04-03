# Hog svm sign dectector


Description: THe Hog-SVM dectects road signs in images. This is down by using a sliding window over each image. creates smaller images "windows" to be classified.
A trained Machine learning Classifier is then used to identify if there is a sign in the window.




![image](https://user-images.githubusercontent.com/45408401/113484438-2e748b00-94a0-11eb-8cbb-58fc77a02779.png)


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
