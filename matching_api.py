#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:36:34 2018

@author: Soumya
"""


import cv2
import numpy as np


img_height = 923
img_width = 600

def load_and_resize(original_path, annotated_path):
    """
    Function: 
        Function is responsible for loading the images and resizing them.
    
    Parameters:
        original_path  : The path to the original image file
        annotated_path : The path to the annotated image file
        
    Returns:
        This function returns the original and annotated images
        of 1200x600 size as nd-numpy arrays
    """
    # Loading images in grayscale.
    original_img = cv2.imread(original_path,0)
    annotated_img = cv2.imread(annotated_path,0)
    
    # Resize to 1200X600 length X width 
    original_img = cv2.resize(original_img, (img_width,img_height)) 
    annotated_img = cv2.resize(annotated_img, (img_width,img_height))

    return original_img, annotated_img


def orb_flann_matcher(original_img,annotated_img):
    """
    Function: 
        Function is responsible for taking two images as input,
        finding the features in them using ORB,
        and finally finding the match in the target image.
    
    Parameters:
        original_img  : The original image file.
        annotated_img : The annotated image file.
        
    Returns:
        This function returns coordinates of the area in the annotated 
        image that matches the central text in the original image. 
    """
        
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
    
    
    #res = cv2.drawMatches(original_img, keypoints_original, 
    #                      annotated_img, keypoints_annotated, 
    #                      good_points[:300], None)
    
    # Homography
    if len(good_points)>10:
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
        
        '''
        cv2.polylines(annotated_img,
                      [np.int32(transform_coordinates)],
                      True, 
                      (255,0,0), 
                      3)
        '''
        
        #cv2.imshow('annotated',annotated_img)
        #cv2.imshow('result',res)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        
        transform_coordinates = np.int32(transform_coordinates)
        return transform_coordinates
    else:
        print('Not enough good matches!')
        return None
    
    
def orb_bf_matcher(original_img,annotated_img):
    """
    Function: 
        Function is responsible for taking two images as input,
        finding the features in them using ORB,
        and finally finding the match in the target image
        using the brute-force matcher.
    
    Parameters:
        original_img  : The original image file.
        annotated_img : The annotated image file.
        
    Returns:
        This function returns coordinates of the area in the annotated 
        image that matches the central text in the original image. 
    """
        
    # ORB
    orb = cv2.ORB_create(nfeatures = 2000)
        
    # Detecting keypoints and calculating the descriptor 
    keypoints_original, descriptors_original = orb.detectAndCompute(original_img, None)
    keypoints_annotated, descriptors_annotated = orb.detectAndCompute(annotated_img, None)
    
    # Matching using BruteForce matcher 
    # Hamming distance as comparison criteria
    # If crossCheck is true, Matcher returns only those 
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
        
        #cv2.imshow('annotated',annotated_img)
        #cv2.imshow('result',res)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
    
        transform_coordinates = np.int32(transform_coordinates)
        return transform_coordinates
    
    else:
        print('Not enough good matches!')
        return None


def surf_flann_match(original_img,annotated_img):
    """
    Function: 
        Function is responsible for taking two images as input,
        finding the features in them using SURF,
        and finally finding the match in the target image.
    
    Parameters:
        original_img  : The original image file.
        annotated_img : The annotated image file.
        
    Returns:
        This function returns coordinates of the area in the annotated 
        image that matches the central text in the original image. 
    """
    
    # SURF 
    surf = cv2.xfeatures2d.SURF_create()
    
    keypoints_original_img, descriptor_original_img = surf.detectAndCompute(original_img,None)
    
    keypoints_annotated_img, descriptor_annotated_img = surf.detectAndCompute(annotated_img,None) 
    
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
    

def detect_extra_text(annotated_img,transform_coordinates):
    """
    Function: 
        Function is responsible for detecting the extra text present
        in the annotated image.
    
    Parameters:
        annotated_img : The annotated image file.
        transform_coordinates : The coordinates of the matched part 
    Returns:
        This function returns contours of the extra text detected in 
        the image. 
    """
    
    # Thresholding the annotated image to remove noise
    _,new_annotated_img = cv2.threshold(annotated_img,
                                        80,
                                        255,
                                        cv2.THRESH_BINARY)
    
    # Fills the ploygon created by the transforma_coordinates 
    cv2.fillPoly(new_annotated_img,
                 [np.int32(transform_coordinates)],
                 255)
    
    new_annotated_img = cv2.Canny(np.asarray(new_annotated_img),100, 200)
    
    
    # 3X3 Matrix will be used as kernel
    N =3
    kernel = np.zeros((N,N),dtype=np.uint8)
    
    # Creating a '-' sign kernel    
    kernel[int((N-1)/2),:] = 1
    
    # Dilating the image    
    # If we dilate for more iterations, bounding 
    # boxes will be paragraph level, if iterations = 4 then word level
    new_annotated_img = cv2.dilate(new_annotated_img, 
                             kernel, 
                             iterations = 25)
        
    # Finding contours
    _, contours, _ = cv2.findContours(new_annotated_img, 
                                      cv2.RETR_TREE, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    return contours


def counters_data_list(contours, img):
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


def detect_underline(annotated_img, transform_coordinates):
    """
    Function: 
        Function is responsible for detecting the underlines present
        in the annotated image.
    
    Parameters:
        annotated_img : The annotated image file.
        
    Returns:
        This function returns contours of the extra text detected in 
        the image. 
    """
    '''
    x,y,w,h = cv2.boundingRect(transform_coordinates)
    
    #extracted_img = annotated_img[y:y+h,x:x+w]
    height, width = annotated_img.shape
    annotated_img[0:y,0:x] = 255
    annotated_img[h:height,w:width] = 255
    cv2.imshow('white', annotated_img)
    '''
    
    # Thresholding
    _,new_img = cv2.threshold(annotated_img,127,255,cv2.THRESH_BINARY)
    
    # Use transform coordinates and remove the extra text part of the
    # annotated image
    
    # Creating white mask by filling the matched area with white
    mask = cv2.fillPoly(new_img, 
                        [np.int32(transform_coordinates)], 
                        255)
    # XOR will make text pixels white and outer text black 
    new_img = cv2.bitwise_xor(annotated_img, mask)
    
    # Thresholding again removes the extra text keeping only required 
    # central text
    _,new_img = cv2.threshold(new_img,127,255,cv2.THRESH_BINARY)
    
    # Running edge detection
    new_img = cv2.Canny(np.asarray(new_img),100, 200)
    
    # Finding contours    
    _, contours, _ = cv2.findContours(new_img, 
                                      cv2.RETR_TREE, 
                                      cv2.CHAIN_APPROX_SIMPLE)
    
    # Getting area, heigh and width lists
    al,wl,hl = counters_data_list(contours, new_img)
            
    mask = np.zeros(new_img.shape, dtype="uint8")
    for i,c in enumerate(contours):
        if  wl[i] > 21:
            # Drawing the bad contour
            new_img = cv2.drawContours(mask, [c], -1, (255), -1)
            # Masking the bad contour
            new_img = cv2.bitwise_and(new_img,
                                      new_img,
                                      mask=mask)
            
            
    # Get the new contours. This will give us the underlines
    _, contours,_= cv2.findContours(new_img, 
                                    cv2.RETR_TREE, 
                                    cv2.CHAIN_APPROX_SIMPLE)
    return contours
            


def box_center(box):
    """
    Function: 
        Function finds the center of the box.
    
    Parameters:
        box      : coordinates of the box.
        
    Returns:
        This function returns the coordinates of 
        the center of the box.
    """
    
    x1,y1 = box[1]
    x2,y2 = box[3]
    
    x_center = (x1+x2)/2
    y_center = (y1+y2)/2
    
    return x_center, y_center


def box_vector(box,width,height,box_type):
    """
    Function: 
        Function is responsible for finding center and normalizing 
        box parameters, builds the box vector required for training.
        [<class-prediction> <x> <y> <width> <height>].
    
    Parameters:
        box      : coordinates of the box.
        width    : width of the box.
        height   : height of the box.
        box_type : what kind of annotation does it enclose.
        
    Returns:
        This function returns normalized vector which will be used 
        with the image to train the model.
    """
    # Finding the center coordinates of the box
    x_center, y_center = box_center(box)
    
    # Normalizing 
    
    # Experimental size, make config file and load
    image_width = img_width
    image_height = img_height
    
    x_center = x_center / image_width
    y_center = y_center / image_height
    
    width = width / image_width
    height = height / image_height
    
    if box_type == 'ext':
        class_pred = 0
    elif box_type == 'underline':
        class_pred = 1
    else:
        print("Class not yet added, returning None as vector")
        return None
    # Note in the final label vector used for training 0 = no class
    
    # Packaging and returning the vector
    return (class_pred, x_center, y_center, width, height)
    
    




