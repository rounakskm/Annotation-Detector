#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 16:10:31 2018

@author: Soumya
"""

'''
Steps:
1. Load and resize images. 
   
2. Crop the original image to get the central printed text region. 
   This operation removes the white parts on the sides of the page and
   help in detecting the central printed text in the annotated image.
   Irrespective of scaling, rotation etc.
   
3. Using ORB along with Flann matcher we first detect the features in both
   images and then match the images, finding the coordinates of the matched 
   region in the annotated image.
   
4. Using the transform coordinates we sunstract the common area and then,
   obtain the extra text. The extra text in the image is dilated to find
   smoother and larger components. The dect_extra_text function returns 
   the contours of these components. As of this step, extra text
   detection is complete, we only need to draw them.
   
5. We now perform underline and between the line annotation detection.
   This is done by taking the annotated image performing edge detection 
   on it, finding contours and then discarding contours with less width.
 
6. Create the vector for each bounding box in the image, and write it to 
   a file with the same name as the image file. Save both the files in
   dataset directory.
   
7. Finally we draw bounding boxes around the extra text and underlines found.
   Save them in another directory for vizualization.   
'''

# Importing required libraries and files
import os
import cv2
import copy
import argparse
import numpy as np
import matching_api as api
from crop_original import crop_img



parser = argparse.ArgumentParser()

parser.add_argument("-o", "--original-img", type=str,
            help="please enter the path to the original image with extension")

parser.add_argument("-a", "--annotated-img", type=str,
            help="please enter the path to the annotated image with extension")

args = parser.parse_args()



original_img_path = args.original_img
annotated_img_path = args.annotated_img


# Loading the images
original_img, annotated_img = api.load_and_resize(original_img_path, 
                                                  annotated_img_path)

# Cropping and obtining the central text parts from the image
cropped_original = crop_img(original_img)
cv2.imshow('croped',cropped_original)
# Finding the features and matching the images
transform_coordinates = api.orb_flann_matcher(cropped_original,annotated_img)

# Find extra text using the match detected
extra_text_contours = api.detect_extra_text(annotated_img,
                                            transform_coordinates)

# Find the underlines
underline_contours = api.detect_underline(annotated_img, 
                                          transform_coordinates)

# Convert the images into color
annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_GRAY2BGR)

# Make a copy of the annotated image
copy_img = copy.copy(annotated_img)


# Writing annotated image to disk(no boxes drawn on it)
# Code to auto increment file number and save the new detected 
# image in the data directory and write the file.

list = os.listdir('dataset') # dataset is the directory used to save new images
number_files = len(list)//2

file_name_img = str(number_files+1)+'.jpg'

# Writing the new image to the data folder
detected_img_path = 'dataset/'+file_name_img
cv2.imwrite(detected_img_path,annotated_img)


# Extra text bounding boxes
for cnt in extra_text_contours:
    rect = cv2.minAreaRect(cnt)
    width,height = rect[1]
    area = width * height
    # Having an area threshold to eleminate 
    # Ink blot and other tiny detections
    if area < 500:
        pass
    else:
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        box_type = 'ext' # As this is extra text
        box_vector = api.box_vector(box,width,height,box_type)
        box_vector = str(box_vector)
        box_vector = box_vector[1:-1]
        box_vector = box_vector.replace(',','')
        
        # Write box_vector in one line of the file
        # File must have same name as the image
        file_name_text = 'dataset/'+str(number_files+1)+'.txt'
        
        with open(file_name_text, 'a') as file:
            file.write(box_vector+'\n')

        # This drawing wont be necessary as CNN needs the clear image
        # and coordinates of class or box, 
        # On for now to visualize the boxes (BLUE boxes)
        annotated_img = cv2.drawContours(annotated_img, [box], 0,(255,0,0),2)

# Underline bounding boxes
for cnt_u in underline_contours:
    rect = cv2.minAreaRect(cnt_u)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    '''
    # For now only detecting external annotations.
    # So input label-vector will have only one classs
    
    box_type = 'underline' # As this is underline
    box_vector = api.box_vector(box,width,height,box_type)
    
    # Write box_vector in one line of the file
    # File must have same name as the image
    file_name_text = 'dataset/'+str(number_files+1)+'.txt'
    
    with open(file_name_text, 'a') as file:
        file.write(str(box_vector)+'\n')
        '''
    # This drawing wont be necessary as CNN needs the clear image
    # and coordinates of class or box, 
    # On for now to visualize the boxes (RED boxes)
    annotated_img = cv2.drawContours(annotated_img, [box], 0,(0,0,255),2)
    
# Displaying the images
cv2.imshow('annotated',annotated_img)
cv2.imshow('copy',copy_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Writing box drawn annotated image to another directory
# Code to auto increment file number and save the new detected 
# image in the data directory and write the file.

list = os.listdir('data') # data is the directory used to save new images
number_files = len(list)

file_name = str(number_files+1)+'.jpg'

# Writing the new image to the data folder
detected_img_path = 'data/'+file_name
cv2.imwrite(detected_img_path,annotated_img)

