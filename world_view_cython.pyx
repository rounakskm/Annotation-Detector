#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 19 15:04:21 2018

@author: Soumya
"""
import cython

# Use both imports else gives error.
cimport numpy as np
import numpy as np


# Function to extract coordinates of the point

#@cython.boundscheck(False)
cpdef list extract_coordinates_fast(unsigned char [:, :] img):
  cdef int i,j,h,w
  
  h = img.shape[0]
  w = img.shape[1]
  
  l_append = list.append
  cdef list point_set=[]

  
  for i in range(0,h):
      for j in range(0,w):
          if img[i][j] == 0:
              l_append(point_set, (j,i))
  return point_set #list of (x,y) coordinates in image plane(top-left is 0,0)


# Function to calculate eucliedian distance
cdef double euclid_dist(tuple point1, tuple point2):
   return ((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)**0.5




# Function to build the world_view_vector
   
cpdef list build_world_view(list point_set):

    cdef double CONV_ANG = (180.0/3.14)
  
    cdef list list_wv = [] # list of world views
    cdef double max_distance = 0.0
    cdef double min_distance = 1e10
    
    l_append = list.append
    
    cdef int no_points 
    cdef int i,j =0
    cdef double diff_x, diff_y
    cdef double distance
    cdef double ang
    #cdef tuple tup
    #cdef list wv
    
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
                    
            #if distance < 0.0:
             #   print(f"build_world_view: i={i}, j={j}, dis={distance}")
            #if ang < 0.0 or ang > 360.0:
             #   print(f"build_world_view: i={i}, j={j}, ang={ang}")
    
            # Add distance, angle pair to the world-view vector here
            # This is the world view of i looking at all other points
            #tup = (distance,ang) 
            l_append(wv,(distance,ang))
            
        # Add WV vector to the lis of vectors here
        # This will have world-view vectors of all the points in the image
        l_append(list_wv, wv)
        
    return list_wv
    
# Things to do:
        # Find good way to compare all the vectors in the list of WV
        # We will build two such lists and the goal is to compare 
        # each vector in list_wv1 with each vector in list_wv2