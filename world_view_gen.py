#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  19 00:28:45 2018

@author: Soumya
"""

import cv2
import time
import math
import numpy as np

import pyximport; pyximport.install()

# Read image as gray-scale
img1 = cv2.imread('img1.jpg',0)
img2 = cv2.imread('img2.jpg',0)

# Resizing the image width X length
img1 = cv2.resize(img1,(175,300))
img2 = cv2.resize(img2,(175,300))


# Thresholding removes background noise like dust particles and small lines 
#  that appear while scanning the physical document
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


# Function to calculate euclidean distance between two points
def euclid_dist(point1, point2):
   return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2) 

# Function to build the world-view vector
# no_points = len(point_set)

CONV_ANG = (180.0/math.pi)

def build_world_view(point_set):
    
    list_wv = [] # list of world views
    max_distance = 0.0
    min_distance = 1e10
    
    l_append = list.append
    
    no_points = len(point_set)
    for i in range(no_points):
        wv = [] # one world-view for each point
        for j in range(no_points):
            distance = euclid_dist(point_set[i],point_set[j])
            diff_x = point_set[j][0] - point_set[i][0]
            diff_y = point_set[j][1] - point_set[i][1]
            
            if abs(distance)>1e-10:
                if diff_x >= 0.0:
                    ang = np.arcsin(diff_y/distance)
                else:
                    ang = np.pi - np.arcsin(diff_y/distance)
            else:
                ang = 0.0
                
            ang *= CONV_ANG
            
            if ang < 0.0:
                ang += 360.0
            else:
                if ang > 360.0:
                    ang -= 360.0
            
            if distance > max_distance:
                max_distance = distance
            else:
                if distance < min_distance:
                    min_distance = distance
                    
            if distance < 0.0:
                print(f"build_world_view: i={i}, j={j}, dis={distance}")
            if ang < 0.0 or ang > 360.0:
                print(f"build_world_view: i={i}, j={j}, ang={ang}")
    
            # Add distance, angle pair to the world-view vector here
            # This is the world view of i looking at all other points
            l_append(wv,(distance,ang))
            
        # Add WV vector to the lis of vectors here
        # This will have world-view vectors of all the points in the image
        l_append(list_wv, wv)
        
    return list_wv
    
# Things to do:
        # Find good way to compare all the vectors in the list of WV
        # We will build two such lists and the goal is to compare 
        # each vector in list_wv1 with each vector in list_wv2

l_wv = build_world_view(ps1)    
l_wv2 = build_world_view(ps2)

    





