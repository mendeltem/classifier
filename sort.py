#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 23:55:08 2019

@author: pandoora
"""
import os
import numpy as np
import imageio
import tensorflow as tf
import matplotlib.pyplot as plt

from lib import load_images,binary_list_converter,load_image,rounding
from lib import get_img_paths, image_resize,resize_height
from models import classifier


keras = tf.keras
models = keras.models

load_model = keras.models.load_model
model_loaded = load_model('boob_classifier.h5')

height_for_model = 320
width_for_model = 320

height  = 1200
width   = 1200

img_dir = 'raw_images'
output_path = "output/boobs"


img_dirs = get_img_paths(img_dir)

list_of_unique_classes = []

for i,image_dir in enumerate(img_dirs):
    
    try:
        new_img = load_image(image_dir, height,width)
        new_img.shape
        image_input = load_image(image_dir, height_for_model,width_for_model)
        image_input.shape
        
        image_input = image_input.reshape( (1,) + image_input.shape)    
        image_input.shape
        
        if len(image_input.shape) == 4 and image_input.shape[-1] == 3:
            predicted = model_loaded.predict(image_input)
            
            predicted_class = binary_list_converter(np.array([rounding(i, 0.9) for i in predicted[0]]).astype(int))
            
            if predicted_class not in list_of_unique_classes:
                list_of_unique_classes.append(predicted_class)
                if not os.path.exists(output_path+"/"+str(predicted_class)+"/"):
                    os.makedirs(output_path+"/"+str(predicted_class)+"/")
                    
            for z,class_p in enumerate(list_of_unique_classes):
                
                if predicted_class == class_p:
                    imageio.imwrite(output_path+"/"+str(predicted_class)+"/" +str(i)+".jpg",new_img) 
    except:
        print("Error")    


