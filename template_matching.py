#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 20:05:10 2018

@author: rounak
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('img1_crop.jpg',0)
img2 = cv2.imread('img2.jpg',0)

# Resizing the image width X length
#img1 = cv2.resize(img1,(600,1200))
#img2 = cv2.resize(img2,(600,1200))

# Thresholding
th_val1,img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
th_val2,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)

res = cv2.matchTemplate(img2, img1, cv2.TM_CCOEFF_NORMED)

# Showing images
cv2.imshow('img2',img2)
cv2.imshow('img1',img1)
cv2.imshow('res',res)
cv2.waitKey(0)
cv2.destroyAllWindows()

