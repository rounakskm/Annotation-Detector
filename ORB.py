#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:43:33 2018

@author: Soumya
"""

import cv2
from crop_original import crop_img
import numpy as np

# Reading the images
original_img = cv2.imread('img1.jpg',0)
annotated_img = cv2.imread('img2_underlined.jpg',0)

original_img = cv2.resize(original_img, (600,1200)) 
annotated_img = cv2.resize(annotated_img, (600,1200))

original_img = crop_img(original_img)

# Transformations and matching

# SIFT
# sift = cv2.xfeatures2d.SIFT_create()

# SURF
# surf = cv2.xfeatures2d.SURF_create()

def find_match(original_img,annotated_img):
        
    # ORB
    orb = cv2.ORB_create(nfeatures = 2000)
        
    # Detecting keypoints and calculating the descriptor 
    keypoints_original, descriptors_original = orb.detectAndCompute(original_img, None)
    keypoints_annotated, descriptors_annotated = orb.detectAndCompute(annotated_img, None)
    
    # Matching using BruteForce matcher 
    # Hamming distance as comparison criteria
    # If it is true, Matcher returns only those 
    # matches with value (i,j) such that i-th descriptor 
    # in set A has j-th descriptor in set B as the 
    # best match and vice-versa.
    # Provides consistent results
    bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck = True)
    
    matches = bf.match(descriptors_original, descriptors_annotated)
    
    matches = sorted(matches, key = lambda x:x.distance)
    
    #res = cv2.drawMatches(original_img, keypoints_original, 
    #                      annotated_img, keypoints_annotated, 
    #                      matches[:300], None)
    
    
    
    # Homography
    if len(matches)>50:
        query_pts = np.float32([keypoints_original[m.queryIdx].pt 
                                for m in matches]).reshape(-1,1,2)
        train_pts = np.float32([keypoints_annotated[m.trainIdx].pt 
                                for m in matches]).reshape(-1,1,2)
    
        # RANSAC algo used to match feature points and estimate parameters    
        matrix,_ = cv2.findHomography(query_pts,train_pts,cv2.RANSAC, 5.0)
        
        #matches_mask = mask.ravel().tolist()
        
        # Perspective transform
        h,w = original_img.shape
        # Getting end coordinates of the original image
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        # Getting the respective coordinates for those points on
        # the annotated image. 
        # Here matrix is the hemography matrix computed above.
        transform_coordinates = cv2.perspectiveTransform(pts, matrix)
        
        
        cv2.polylines(annotated_img,
                      [np.int32(transform_coordinates)], 
                      True, 
                      (255,0,0), 
                      3)
        
        cv2.imshow('annotated',annotated_img)
        #cv2.imshow('result',res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    
        transform_coordinates = np.int32(transform_coordinates)
        return transform_coordinates
    

transform_coordinates = find_match(original_img, annotated_img)
mask = np.zeros((annotated_img.shape[0], annotated_img.shape[1]))

cv2.fillConvexPoly(mask, transform_coordinates, 1)
mask = mask.astype(np.bool)

out = np.zeros_like(annotated_img)
out[mask] = annotated_img[mask]

cv2.imshow('out',out)
cv2.waitKey(0)
cv2.destroyAllWindows()



