#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 01:14:28 2018

@author: Soumya
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Loading images
img1 = cv2.imread("img1.jpg")
img2 = cv2.imread("img2.jpg")

# Resizingg images
img1 = cv2.resize(img1, (600,1200)) 
img2 = cv2.resize(img2, (600,1200))


x = 90
y = 55
width = 550 - 90
height = 1065 - 55

# Thresholding
th_val1,img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
th_val2,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)


# Converting from rgb to hsv
hsv_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
hsv_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

# Only using one channel (hue)
img1_hist = cv2.calcHist([hsv_img1], [0], None, [180], [0, 180])

# All values must be in range 0-255 so normalize
img1_hist =cv2.normalize(img1_hist,img1_hist, 0, 255, cv2.NORM_MINMAX)

# Creating the mask for img2 by Backprojection
mask = cv2.calcBackProject([hsv_img2], [0], img1_hist, [0,180], 1)


# Making and drawing bounding box
term_criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
_, window = cv2.meanShift(mask, (x,y,width,height), term_criteria)

x,y,w,h = window

cv2.cv2.rectangle(img2, (x,y), (x+w,y+h), (0,255,0))

# Showing images
cv2.imshow('img2',img2)
cv2.imshow('img1',img1)
cv2.imshow('mask',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

