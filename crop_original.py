#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 14:17:04 2018

@author: Soumya
"""

import cv2
import numpy as np


def dilate_image(img, N, iterations):
    """
    Function: 
        This function is responsible for dilating the image
    
    Parameters:
        img        : The imput image
        N          : Dimension of the + sign shape (NxN)
        iterations : Number of iterations
        
    Returns:
        This function returns the dilated image.
    """
    kernel = np.zeros((N,N),dtype=np.uint8)
    
    kernel[int((N-1)/2),:] = 1
    
    dilated_img = cv2.dilate(img/255, kernel, iterations = iterations)
    
    kernel = np.zeros((N,N),dtype=np.uint8)
    kernel[:,int((N-1)/2)] = 1
    dilated_img = cv2.dilate(dilated_img, kernel, iterations = iterations)
    
    return dilated_img

def find_components(edges):
    """
    Function: 
        Finds the text components in the image by performing 
        aggressive dilation until there are only a few connected components
    
    Parameters:
        edges    : The edge detected image given by the Canny edge detector
        
    Returns:
        This function returns the contours of the text 
        components found in the image.
    """
    count = 20
    n = 1
    
    # Aggressive dilation will make all the central text 
    # into one or two components
    while count > 15:
        n += 1
        dilated_img = dilate_image(edges, N=3, iterations=n)
    
        # As the image is a an array of values between 0.0 to 1.0
        # we need to make all values between 0 to 255
        dilated_img = np.uint8(dilated_img * 255)    
        # Finding contours
        _, contours, _ = cv2.findContours(dilated_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Number of contours will keep decreasing with dilation
        # count value will decide number of contours left
        # terminate loop few contours left (1 if central text is all merged)
        count = len(contours)
        
    # Displaying the dilated image
    #cv2.imshow('dilated_img',dilated_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return contours

def bounding_box_counters(contours, img):
    """
    Function: 
        Finds a bounding box and number of set pixels for each contour.
    
    Parameters:
        contours : The contours of the dilated text components
        img      : The input image
        
    Returns:
        This function returns the a list of dictionaries with info 
        about the coordinates of the bounding box and set of pixels.
    """
    info = []
    
    for contour in contours:
        x,y,width,height = cv2.boundingRect(contour)
        c_im = np.zeros(img.shape)
        cv2.drawContours(c_im, [contour], 0, 255, -1)
        info.append({
                'x1':x,
                'y1':y,
                'x2':x+width -1,
                'y2':y+height -1,
                'sum': np.sum(img*(c_im>0))/255
                })
    return info

def calculate_crop_area(crop):
    """
    Function: 
        Finds the area of the crop provided
    
    Parameters:
        crop : A tuple containing 4 coordinates
        
    Returns:
        This function returns a area of the crop selected.
    """
    x1,y1,x2,y2 = crop
    return max(0,x2-x1) * max(0,y2-y1)


def crop_union(crop1, crop2):
    """
    Function: 
        Finds the union of two crop rectangles
    
    Parameters:
        crop1 : A tuple containing 4 coordinates of a crop rectangle.
        crop2 : A tuple containing 4 coordinates of a crop rectangle.
        
    Returns:
        This function returns the union of two rectangles.
    """
    x11,y11,x21,y21 = crop1
    x12,y12,x22,y22 = crop2
    return min(x11,x12), min(y11,y12), max(x21,x22), max(y21,y22)

def find_optimal_components(contours, edges, img):
    """
    Function: 
        Finds a crop that is balanced in terms of 
        coverage/compactness.
        Solves the precision/recall tradeoff problem.
        Uses greedy approach and keeps adding components till F1 score 
        is increasing. And stops when adding a component reduces F1 score.
    
    Parameters:
        contours : The contours of the dilated text components
        edges    : The edge detected image given by the Canny edge detector
        img      : The original resized image
    Returns:
        This function returns a crop for the edge image entered.
    """
    info = bounding_box_counters(contours, img)
    info.sort(key=lambda x: -x['sum'])
    total = np.sum(edges)/255
    area = edges.shape[0] * edges.shape[1]
    
    c = info[0]
    del info[0]
    crop_instance = c['x1'], c['y1'], c['x2'], c['y2']
    crop = crop_instance 
    covered_sum = c['sum']
    
    while covered_sum < total:
        flag = False
        recall = 1.0 * covered_sum/total
        precision = 1 - 1.0 * calculate_crop_area(crop) / area
        # Calculating F1 score
        f1 = 2 * (precision * recall / (precision + recall))
        
        for i, c in info:
            crop_instance = c['x1'], c['y1'], c['x2'], c['y2']
            crop_new = crop_union(crop, crop_instance)
            new_sum = covered_sum + c['sum']
            new_recall = 1.0 * new_sum / total
            new_precision = 1 - 1.0 *  calculate_crop_area(crop_new) / area
            new_f1 = 2 * (new_precision * new_recall / (new_precision + new_recall))
            
            # Add this crop, if it increases f1 score or 
            # adds 25% of remaining pixels with less than 
            # 15% crop area expansion
            remaining_frac = c['sum'] / (total-covered_sum)
            new_area_frac = 1.0 * calculate_crop_area(crop_new) / calculate_crop_area(crop) -1
            
            if (new_f1 > f1) or (remaining_frac > 0.25 and new_area_frac < 0.15):
                crop = crop_new
                covered_sum = new_sum
                del info[i]
                flag = True
                break
            
        if not flag:
            break

    return crop


def crop_img(img):
    """
    Function: 
        Corps the given image, closest to the text 
    
    Parameters:
        img : The input image
        
    Returns:
        This function returns the cropped image of the given original 
        image, cropped keeping only the central textul part.
    """
    # Thresholding the image to remove noise
    _,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)

    # Finding edges
    edges = cv2.Canny(np.asarray(img),100, 200)
    
    edges = 255 * (edges > 0).astype(np.uint8)

    # Finding components (treating each unit of text as a component)
    contours = find_components(edges)     

    if len(contours) == 0:
        print("No text in the image")
    
    # Compute the crop coordinates    
    crop = find_optimal_components(contours, edges, img)
    
    # Crop coordinates
    x1,y1,x2,y2 = crop
    
    # Crop the original_img
    # splice row-range considered, colmn-range considered
    img = img[y1:y2, x1:x2]
    return img
    
    

