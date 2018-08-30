#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:07:53 2018

@author: rounak

"""

import cv2
import matplotlib.pyplot as plt

#img =  cv2.imread('img1.jpg')
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#cv2.imshow('image',gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


#plt.imshow(gray)

#saving image
#cv2.imwrite('gray.jpg', gray)


# Read image as gray-scale
img1 = cv2.imread('img1.jpg')
img2 = cv2.imread('img2.jpg')

# Resizing the image width X length
img1 = cv2.resize(img1,(300,600))
img2 = cv2.resize(img2,(300,600))

#img1 = cv2.resize(img1,(175,300))
#img2 = cv2.resize(img2,(175,300))

# Thresholding removes background noise like dust particles and small lines 
# that appear while scanning the physical document
# threshold(src.img, threshold-value, max_val, thresholding_type)

th_val1,img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
th_val2,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)


blue=(255,0,0)
img1 = cv2.cv2.line(img1, (0,0), (300,600), blue, thickness = 5)

plt.imshow(img1)




cv2.imshow('image',img1)
cv2.waitKey(0)
cv2.destroyAllWindows()



# Capturing video
cap = cv2.VideoCapture(0)

# Loading a file
#cap = cv2.VideoCapture('path of file .extension')

while True:
    ret, frame = cap.read()
    
    cv2.imshow('frame',frame)
    
    
    #key = cv2.waitKey(30) will play loaded video at 30 fps
    key = cv2.waitKey(1)
    if key == 27:
        break
    
cap.release()
cv2.destroyAllWindows()



