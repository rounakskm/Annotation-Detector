#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:09:02 2018

@author: rounak
"""

import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import pyximport; pyximport.install()

# Read image as gray-scale
img1 = cv2.imread('img1.jpg',0)
img2 = cv2.imread('img2.jpg',0)

# Resizing the image width X length
img1 = cv2.resize(img1,(1750,3000))
img2 = cv2.resize(img2,(1750,3000))

#img1 = cv2.resize(img1,(175,300))
#img2 = cv2.resize(img2,(175,300))

# Thresholding removes background noise like dust particles and small lines 
# that appear while scanning the physical document
# threshold(src.img, threshold-value, max_val, thresholding_type)

th_val1,img1 = cv2.threshold(img1,127,255,cv2.THRESH_BINARY)
th_val2,img2 = cv2.threshold(img2,127,255,cv2.THRESH_BINARY)


import world_view_cython

start1 = time.time()
# Extracting the point set
ps1 = world_view_cython.extract_coordinates_fast(img1)
ps2 = world_view_cython.extract_coordinates_fast(img2)
end1 = time.time()

print(f'Time taken to extract coordinates from two 175X300 images: {end1-start1}')
#print(f'Time taken to extract coordinates from one 1750X3000 image: {end1-start1}')
# Takes 0.15 seconds for one 1750X3000 image to be processed

'''
start2 = time.time()
#Creating world_view for point-set1
l_wv = world_view_cython.build_world_view(ps1)
end2 = time.time()

print(f'Time taken to create world_view for one point-set: {end2-start2}')
'''

# Benchmark 

# Only using static declaration with cython
# Time taken to extract coordinates from two 175X300 images: 0.0029990673065185547
# Time taken to create world_view for one point-set: 17.174129247665405

# Now using Multi-Processing
# For multi-processing to work, we need to make sure that no python is used
# after the GIL is released. So nogil will check if python is used. Code needs 
# to be pure cython.


'''
# Python for multiprocessing:
from concurrent.futures import ThreadPoolExecutor

start2 = time.time()
#Creating world_view for point-set1
with ThreadPoolExecutor (max_workers= 4) as exe:
    sections = np.array_split(ps1, 4) # Splits ps1 into 4 views
    jobs = [exe.submit(world_view_cython.build_world_view, s) for s in sections]
    
sum(job.result() for job in jobs)

print(f'Time taken to create world_view for one point-set: {end2-start2}')
'''