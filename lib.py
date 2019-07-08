#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:57:17 2019

@author: pandoora
"""
import os
import cv2
import math
import numpy as np
import imageio

seed = 42

def get_img_paths(start_dir, extensions = ['png','jpeg','jpg','pneg']):
    """Returns all image paths with the given extensions in the directory.

    Arguments:
        start_dir: directory the search starts from.

        extensions: extensions of image file to be recognized.

    Returns:
        a sorted list of all image paths starting from the root of the file
        system.
    """
    if start_dir is None:
        start_dir = os.getcwd()
    img_paths = []
    for roots,dirs,files in os.walk(start_dir):
        for name in files:
            for e in extensions:
                if name.endswith('.' + e):
                    img_paths.append(roots + '/' + name)
    img_paths.sort()
    return img_paths


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def resize_height(image , height):
    """resize the height of a picture and the width stays
    it cuts from the picture or adds zeros to the picture
    
    """

    (h, w) = image.shape[:2]
    
    if height < h:
        mid = int(h/2)
        pad = int(height/2)
        new_image = image[mid-pad: mid+pad]    
    elif height > h:
        new_image = np.zeros((height,w,3))
        mid = math.ceil(h/2)
        pad = math.ceil(height/2)
        try:
            new_image[pad-mid+1: pad+mid] = image
        except:
            new_image[pad-mid: pad+mid] = image
    elif height == h:
        new_image = image
         
    return new_image /255




def load_images(img_dir,height, width ):
    
    im_dirs = get_img_paths(img_dir)    
    new_images = []
    
    for image in im_dirs:
        try:
            res_image = image_resize(imageio.imread(image), width = width)
            (h, w) = res_image.shape[:2]
            
            if height < h:
                mid = int(h/2)
                pad = int(height/2)
                new_image = res_image[(mid-pad): (mid+pad)] 
                
                
            if height > h:
                new_image = np.zeros((height,w,3))
                mid = math.ceil(h/2)
                pad = math.ceil(height/2)
                try:
                    new_image[pad-mid+1: pad+mid] = res_image / 255
    
                except:
                    new_image[pad-mid: pad+mid] = res_image /255
    
            elif height == h:
                new_image = res_image 
                
                
            new_image = new_image.reshape( (1,) + new_image.shape)    
                
            new_images.append(new_image)  
        except:
            print("Error")
            
    return new_images 

      
def create_image_class(image_dir, height, width, name_of_class, output_path,test_split_ratio= 0.1):

    im_dirs = get_img_paths(image_dir)
    
    images_names_list = [ images for images in im_dirs]
    unque_images_length = len(np.unique(images_names_list))
    splitting_imageindex =  int(unque_images_length * (1- test_split_ratio)) 
    
    np.random.seed(seed)
    np.random.shuffle(images_names_list)
    
    train_list = images_names_list[:splitting_imageindex]
    validate_list = images_names_list[splitting_imageindex:]
     
    images_half = []

    for image in train_list:
        
        try:
            image_res = image_resize(imageio.imread(image), width = width)
            images_half.append(image_res)
        except:
            #print(image)
            print("Error")
   
    training_images=[]
    
    for i in range(len(images_half)):
        new_img = resize_height(images_half[i], height )
        if len(new_img.shape) == 3:
            if new_img.shape[2] == 3:
                training_images.append(new_img)   
            
    images_half = []

    for image in validate_list:
        
        try:
            image_res = image_resize(imageio.imread(image), width = width)
            images_half.append(image_res)
        except:
            #print(image)
            print("Error")
   
    testin_images=[]
    
    for i in range(len(images_half)):
        new_img = resize_height(images_half[i], height )
        if len(new_img.shape) == 3:
            if new_img.shape[2] == 3:
                testin_images.append(new_img)   
    
    #os.chdir(output_path)    
    if not os.path.exists(output_path+"/train/" + name_of_class):
        os.makedirs(output_path+"/train/" + name_of_class)
    
    if not os.path.exists(output_path+"/test/" + name_of_class):
        os.makedirs(output_path+"/test/" + name_of_class)
        
          
    for i, image in enumerate(training_images):
        imageio.imwrite(output_path+"/train/" + name_of_class+str(i)+".jpg",image)  
    
    
    for i, image in enumerate(testin_images):
        imageio.imwrite(output_path+"/test/" + name_of_class+str(i)+".jpg",image)  


def load_image_and_resize(image_dir, height, width):
    """load  a list of images"""

    im_dirs = get_img_paths(image_dir)
    
    images_half = [image_resize(imageio.imread(image), width = width)  for image in im_dirs]
   
    new_images=[]
    
    for i in range(len(images_half)):
        new_img = resize_height(images_half[i], height)
        if new_img.shape[2] ==3:
            new_images.append(new_img)
            
            
    return new_images   

       

def load_image(img_dir, height, width):
    """Load and resize the image
    
    """
    try:
        image = resize_height(image_resize(imageio.
                                          imread(img_dir), width = width), height)
        return image
    except:
        print("Error")
   
    
    
def binary_list_converter(predicted_class):
    """converts a binary list into a number"""
    number = 0
    for i,bit in enumerate(predicted_class):
        number += 2 ** (i*1) * bit 
        
    return number    


def rounding(digit, threshold):
    """rounds a float with a threshold""" 
    k = digit % 1
    f = digit - k
    
    if k >= threshold:
        return (f + 1)
    else:
        return f