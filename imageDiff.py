#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:19:17 2018

@author: Soumya
"""
import cv2
import imutils
from skimage.measure import compare_ssim

#import matplotlib.pyplot as plt

img1 = cv2.imread('img1.jpg')

cv2.imshow('image',img1)


