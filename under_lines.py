#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 13:15:30 2018

@author: Soumya
"""

import cv2
import numpy as np
from crop_original import crop_img


def counters_area_list(contours, img):
    """
    Function: 
        Finds the area covered by the rectangele drawn around the 
        contours.
    
    Parameters:
        contours : The contours of the dilated text components
        img      : The input image
        
    Returns:
        This function returns the a list of area of rectangels drawn 
        around contours.
    """
    area_list = []
    width_list = []
    height_list = []
    
    for contour in contours:
        x,y,width,height = cv2.boundingRect(contour)
        area = width * height
        area_list.append(area)
        width_list.append(width)
        height_list.append(height)
    return area_list,width_list,height_list
        

# Reading the images
extracted_img = cv2.imread('extracted_img.jpg')
img2 = cv2.imread('img2_underlined.jpg')
img2 = cv2.resize(img2, (600,1200))
img = cv2.imread('extracted_img.jpg')
img = crop_img(img)
extracted_img = crop_img(extracted_img)

extracted_img = img2

_,extracted_img = cv2.threshold(extracted_img,127,255,cv2.THRESH_BINARY)

extracted_img = cv2.Canny(np.asarray(extracted_img),100, 200)
    
_, contours, _ = cv2.findContours(extracted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

al,wl,hl = counters_area_list(contours, extracted_img)
        
mask = np.zeros(extracted_img.shape, dtype="uint8")
for i,c in enumerate(contours):
    if  wl[i] > 21:
        extracted_img = cv2.drawContours(mask, [c], -1, (255), -1)
        extracted_img = cv2.bitwise_and(extracted_img, extracted_img, mask=mask)        
        

cv2.imshow('extracted',extracted_img)
cv2.imshow('img',img)
cv2.imshow('2',img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get the new contours
_, contours, _ = cv2.findContours(extracted_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes on contours, in original image
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    img2 = cv2.drawContours(img2, [box], 0,(255),0)
    
    