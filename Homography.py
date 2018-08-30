#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 03:51:05 2018

@author: Soumya
"""

import cv2
import numpy as np
from crop_original import crop_img, find_components

original_img = cv2.imread('img1.jpg', cv2.IMREAD_GRAYSCALE)
annotated_img = cv2.imread('img2_underlined.jpg')


original_img = cv2.resize(original_img, (600,1200)) 
annotated_img = cv2.resize(annotated_img, (600,1200))

final_img = cv2.imread('img2_underlined.jpg')
final_img = cv2.resize(final_img, (600,1200))

original_img = crop_img(original_img)


gray_annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2GRAY)
    
# Thresholding removes background noise like dust particles and small lines 
#  that appear while scanning the physical document
# threshold(src.img, threshold-value, max_val, thresholding_type)

#_,original_img = cv2.threshold(original_img,127,255,cv2.THRESH_BINARY)
#_,annotated_img = cv2.threshold(annotated_img,127,255,cv2.THRESH_BINARY)

#sobely = cv2.Sobel(original_img, cv2.CV_64F, 0, 1)


def find_match(original_img,annotated_img):
    # Features 
    sift = cv2.xfeatures2d.SURF_create()
    
    keypoints_original_img, descriptor_original_img = sift.detectAndCompute(original_img,None)
    
    keypoints_annotated_img, descriptor_annotated_img = sift.detectAndCompute(gray_annotated_img,None) 
    
    # Drawing keypoints on original_img
    #original_img = cv2.cv2.drawKeypoints(original_img, keypoints_original_img, original_img)
    
    # Feature Matching
    index_params = dict(algorithm = 0, trees = 5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    matches = flann.knnMatch(descriptor_original_img,descriptor_annotated_img, k =2)
    
    good_points =[]
    for m,n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    # Displaying matches        
    #img3 = cv2.drawMatches(original_img, keypoints_original_img, 
    #                       gray_annotated_img, keypoints_annotated_img, 
    #                       good_points, gray_annotated_img, flags=2)
    
    # Homography
    if len(good_points)>50:
        query_pts = np.float32([keypoints_original_img[m.queryIdx].pt for m in good_points]).reshape(-1,1,2)
        train_pts = np.float32([keypoints_annotated_img[m.trainIdx].pt for m in good_points]).reshape(-1,1,2)
        
        matrix,_ = cv2.findHomography(query_pts,train_pts,cv2.RANSAC, 5.0)
        
        #matches_mask = mask.ravel().tolist()
        
        # Perspective transform
        h,w = original_img.shape
        pts = np.float32([[0,0], [0,h], [w,h], [w,0]]).reshape(-1,1,2)
        transform_coordinates = cv2.perspectiveTransform(pts, matrix)
        '''
        cv2.polylines(annotated_img,
                      [np.int32(transform_coordinates)], 
                      True, 
                      (255,0,0), 
                      3)
        '''
        #cv2.imshow('annotated_img',annotated_img)
        #cv2.imshow('img3',img3)        
        transform_coordinates = np.int32(transform_coordinates)

        return transform_coordinates
    
transform_coordinates = find_match(original_img,annotated_img)

# Getting points for the rectangle covering cenreal text
x,y,w,h = cv2.boundingRect(transform_coordinates)

# Use this extracted image for underline detection 
extracted_img = annotated_img[y:y+h,x:x+w]

cv2.imshow('extracted',extracted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Make the central text go white, to be able to detect exra handwritten text
new_annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2GRAY)

_,new_annotated_img = cv2.threshold(new_annotated_img,127,255,cv2.THRESH_BINARY)

# Fills the ploygon created by the transformation_coordinates 
cv2.fillPoly(new_annotated_img, 
             [np.int32(transform_coordinates)],
             255)

cv2.imshow('new',new_annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

new_annotated_img = cv2.Canny(np.asarray(new_annotated_img),100, 200)

N =3
kernel = np.zeros((N,N),dtype=np.uint8)
    
kernel[int((N-1)/2),:] = 1
kernel[:,int((N-1)/2)] = 1
    
new_annotated_img = cv2.dilate(new_annotated_img, 
                         kernel, 
                         iterations = 3)

_, contours, _ = cv2.findContours(new_annotated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#dilated_img = cv2.dilate(dilated_img, kernel, iterations = iterations)
    

# Draw bounding boxes on contours, in original image
for cnt in contours:
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    final_img = cv2.drawContours(final_img, [box], 0,(255),0)
    
    
cv2.imshow('final',final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
