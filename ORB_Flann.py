#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 21:30:27 2018

@author: Soumya
"""

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
def find_match(original_img,annotated_img):
        
    # ORB
    orb = cv2.ORB_create(nfeatures = 7000)
        
    # Detecting keypoints and calculating the descriptor 
    keypoints_original, descriptors_original = orb.detectAndCompute(original_img, None)
    keypoints_annotated, descriptors_annotated = orb.detectAndCompute(annotated_img, None)
    
    FLANN_INDEX = 6
    
    index_params = dict(algorithm = FLANN_INDEX,
                        table_number = 6,
                        key_size = 12,
                        multi_probe_level = 1)
    
    search_params = dict(checks = 50)
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors_original,descriptors_annotated,k=2)

    good_points =[]
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    print(len(good_points))
    
    res = cv2.drawMatches(original_img, keypoints_original, 
                          annotated_img, keypoints_annotated, 
                          good_points[:300], None)
    
    # Homography
    if len(good_points)>30:
        query_pts = np.float32([keypoints_original[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
        train_pts = np.float32([keypoints_annotated[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
    
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
        cv2.imshow('result',res)
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





