#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 13 23:55:49 2018

@author: Soumya
"""

# Importing Libraries

import os
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ImgDataset(Dataset):
    def __init__(self, data_dir, transform = None, 
                 img_shape = (600,300), grid_dim = (60,30)):
        
        self.img_shape = img_shape
        self.grid_dim = grid_dim
        self.transform = transform
        
        # Create a grid. Grid doesnt change as image size is same
        self.grid_list, self.col_width, self.row_height = \
        self.create_grid(self.img_shape, self.grid_dim)
        
        # Fetch all files from dataset directory
        self.files = os.listdir(data_dir)
        self.files = [os.path.join(data_dir,f) for f in self.files]

        # Make separate list for images and text files
        self.img_list = [f for f in self.files if '.jpg' in f]
        self.txt_list = [f for f in self.files if '.txt' in f]
        self.label_list = []
        
        for txt_file in self.txt_list:
            # Call function to iterate over grid_list 
            # and generate label vector for the image
            label = self.label_vector_gen(txt_file)
            # Has one-to-one correspondence with img_list
            self.label_list.append(label) 
        

    def __len__(self):
        '''Return size of the dataset'''
        return len(self.img_list)
    
    def __getitem__(self,idx):
        '''Enable indexing on dataset'''
        img = Image.open(self.img_list[idx])  # PIL image
        image = self.transform(img)
        return image, self.label_list[idx]



    def create_grid(self,img_shape,grid_dim):
        '''
        Function: 
            Function creates a grid with cells of equal size.
        
        Parameters:
            img_shape : dimensions of the image (height,width).
            grid_dim  : dimensions of the grid (rows,colms).
            Note: width must be divisible by no. of columns
            and height must be divisible by no. of rows
            
        Returns:
            This function returns a list of top left coordinates of cells, 
            along with the col_width and row_height.
        '''
        height, width = img_shape
        rows,colms = grid_dim
        
        col_width = width//colms
        
        row_height = height//rows
        
        grid_list = []
        
        for c in range(0,width,col_width): # Step size is the col_width
            for r in range(0,height,row_height): # Step size is the row_height
                grid_list.append((c,r))
                
        return grid_list, col_width, row_height


    def label_vector_gen(self,txt_file):
        '''
        Function: 
            Function generates the label vector for each image.
        
        Parameters:
            txt_file  : Name of file name that has the bounding box data.
        Returns:
            This function returns the label vector for each image.
            Note: Each label is of size
            grid_rows x grid_colmn x no.of classes x len of vector
            = len(grid_list) x 1 x 5
        '''
        
        # Open file to read box details
        with open(txt_file, 'r') as file:
            bounding_boxes = file.read().split('\n')
            # Remove last emelent as it is empty
            bounding_boxes = bounding_boxes[:-1]
        
        coordinate_list = [] # Store coordinates of cells with bbox center 
        label = [] # Store the labels for each bounding box
        
        if len(bounding_boxes)>0:
            for box in bounding_boxes:
                box = box.split(' ')
                # Each element of box list has one attribute of the box as str
                # un-normalize them by multiplying width and height
                box_center_x = float(box[1]) * self.img_shape[1]
                box_center_y = float(box[2]) * self.img_shape[0]
                
                # Get the coordinate of the cell containing the box centers
                grid_x = (box_center_x//self.col_width) * self.col_width
                grid_y = (box_center_y//self.row_height) * self.row_height
                coordinate_list.append((grid_x,grid_y))
                
            # Check if point lies in cell and generate the vector
            for cell in self.grid_list:
                if cell in coordinate_list:
                    # Make vector as required
                    idx = coordinate_list.index(cell)
                    box_info = bounding_boxes[idx] # Get corresponding box data
                    box_info = box_info.split(' ')
                    
                    class_pred = float(box_info[0]) 
                    x_center = float(box_info[1])
                    y_center = float(box_info[2])
                    width = float(box_info[3])
                    height = float(box_info[4])
                    
                    label.append((class_pred, 
                                  x_center, 
                                  y_center, 
                                  width, 
                                  height))
                else:
                    # 99 means no value as object doesnt exist in the grid
                    # (class_pred, x_center, y_center, width, height)
                    label.append((99,99,99,99,99))
                    
        elif len(bounding_boxes) == 0:
            label = [(99,99,99,99,99,99)] * (self.grid_dim[0] * self.grid_dim[1])
            
        return label
    
    
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])

transform = transforms.Compose([transforms.Resize((923,00)),
                                transforms.ToTensor(),
                                normalize])
    
train_dataset = ImgDataset('dataset', transform = transform, 
                 img_shape = (923,600), grid_dim = (60,30))

train_dataloader = DataLoader(train_dataset,
                              batch_size=4, 
                              shuffle=True,
                              num_workers=0)
        
            
            
# start annotating from image 111_o
                    
                    
                
                
            
