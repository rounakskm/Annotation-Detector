#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 14:22:17 2018

@author: Soumya
"""
# Import Libraries

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


# Create Convolutional Neural Net to detect and localize annotations
# Net should take an image as input and give out a vector equivalent 
# to the label vector as output.
# Input image shape (h=600,w=600,c=3) in batch: 4x3x600x300

# Net should give 900x5 output as of now
# As a 60x30 grid with 5 features per cell = 900x5
# Input batch will have size batch_size x 900x5

# Using YOLO like architecture
# 9 convolutoion layers and one fully connected layer
# Kernel for convolution = 3x3, stride = 1
# Kernel for MaxPooling = 2x2, stride = 2

class CNN(nn.Module):
  
  def __init__(self):
    super(CNN,self).__init__()
    
    # Convolution layer 1
    self.conv1 = nn.Conv2d(in_channels=3, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn1 = nn.BatchNorm2d(36)
    
    # Convolution layer 2
    self.conv2 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn2 = nn.BatchNorm2d(36)
    
    # Convolution layer 3
    self.conv3 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn3 = nn.BatchNorm2d(36)
    
    # Convolution layer 4
    self.conv4 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn4 = nn.BatchNorm2d(36)
    
    # Convolution layer 5
    self.conv5 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn5 = nn.BatchNorm2d(36)
    
    # Convolution layer 6
    self.conv6 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn6 = nn.BatchNorm2d(36)
    
    # Convolution layer 7
    self.conv7 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn7 = nn.BatchNorm2d(36)
    
    # Convolution layer 8
    self.conv8 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    self.bn8 = nn.BatchNorm2d(36)
    
    # Convolution layer 9
    self.conv9 = nn.Conv2d(in_channels=36, out_channels=36, 
                           kernel_size=3, groups=3)
    
    # Output Layer
    self.fc1 = nn.Linear(in_features=128, out_features=900*5)
    
    
  def forward(self,x):
    
    x= self.bn1(self.conv1(x))
    x= F.max_pool2d(x,kernel_size=2,stride=2)
    x= F.leaky_relu(x,inplace=False)
    
    x= self.bn2(self.conv2(x))
    x= F.max_pool2d(x,kernel_size=2,stride=2)
    x= F.leaky_relu(x,inplace=False)
    
    x= self.bn3(self.conv3(x))
    x= F.max_pool2d(x,kernel_size=2,stride=2)
    x= F.leaky_relu(x,inplace=False)
    
    x= self.bn4(self.conv4(x))
    x= F.max_pool2d(x,kernel_size=2,stride=2)
    x= F.leaky_relu(x,inplace=False)
    
    x= self.bn5(self.conv5(x))
    x= F.max_pool2d(x,kernel_size=2,stride=2)
    x= F.leaky_relu(x,inplace=False)
    
    x= self.bn6(self.conv6(x))
    x= F.max_pool2d(x,kernel_size=2,stride=2)
    x= F.leaky_relu(x,inplace=False)
    
    
    x= self.bn7(self.conv7(x))
    
    x= self.bn8(self.conv8(x))
   
    x= self.conv9(x)
    
    x= self.fc1(x)
    
    
    

def train(epochs):
  model.train()
  
  for epoch in range(epochs):
    
    
    # As dataloader has batch_size = 4
    # We will get one batch of 4 images with each iteration 
    for i, (images_batch, target) in enumerate(train_loader):
        # Convert images_batch and target to pytorch Variable
        images_batch, target = Variable(images_batch),Variable(target)
        optimizer.zero_grad() # Make sure gradients are initially 0
        
        # Converting input to type torch.cuda
        images_batch.requires_grad_()
        images_batch = images_batch.to(device)
      
        # Converting targets to type torch.cuda
        target.requires_grad_()
        target = target.to(device)
      
        out = model(images_batch) # Forward pass
        
        loss = criterion(out, target) # Computing the loss
        loss.backward() # Back-Prop the loss / Backward Pass\
        
        optimizer.step() # Update the gradients
      
        print(f'Batch : {i+1} Loss : {loss}')
        
    print(f'Epoch : {epoch+1} Loss : {loss}')
​