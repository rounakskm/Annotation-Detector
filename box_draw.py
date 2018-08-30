#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 14:22:17 2018

@author: Soumya
"""
import cv2
import numpy as np

original_img = cv2.imread('img2.jpg')
original_img = cv2.resize(original_img, (600,923)) 
resized_img = cv2.resize(original_img, (300,600)) 

box_vector ='0 0.10083333333333333 0.6522210184182016 0.20166666666666666 0.08017334777898158'
box_info = box_vector.split(' ')

class_pred = float(box_info[0]) 
x_center = float(box_info[1])
y_center = float(box_info[2])
width = float(box_info[3])
height = float(box_info[4])

x_center_big = x_center * 600
y_center_big = y_center * 923

height_big = height *923
width_big = width *600

x_center_small = x_center * 300
y_center_small = y_center * 600

height_small = height *600
width_small = width *300

def get_coordinates(x_cent,y_cent,height,width,img_shape):
    # img_shape = (height, width)
    img_height, img_width = img_shape
    
    x_cent = x_cent * img_width
    y_cent = y_cent * img_height
    
    height = height * img_height
    width = width * img_width
    
    x1 = x_cent - (width/2)
    x2 = x_cent + (width/2)
    x3 = x2
    x4 = x1
    
    y1 = y_cent - (height/2)
    y4 = y_cent + (height/2)
    y2 = y1
    y3 = y4
    
    # order specified based on cv2.drawContours requirement
    
    box = np.array([[x3,y3], [x4,y4], [x1,y1], [x2,y2]])
    return box
    
    
box = get_coordinates(x_center,y_center,height,width,(923,600))    
box = np.int0(box)

small_box = get_coordinates(x_center,y_center,height,width,(600,300))
small_box = np.int0(small_box)

# Drawing the boxes
original_img = cv2.drawContours(original_img, [box], 0,(255,0,0),2)
resized_img = cv2.drawContours(resized_img, [small_box], 0,(255,0,0),2)

cv2.imshow('original', original_img)
cv2.imshow('resized', resized_img)

cv2.waitKey(0)
cv2.destroyAllWindows()