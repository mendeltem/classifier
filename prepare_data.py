#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 17:56:40 2019

@author: pandoora
"""

from lib import get_img_paths, image_resize,resize_height,create_image_class
import imageio
import matplotlib.pyplot as plt
import numpy as np

import os
import re


image_dir = 'manuel_sorted/boobs_search/0'
seed = 42
#
output_path = "dataset/boobs"
#
#
name_of_class = "0/"

height = 320
width = 320
#
test_split_ratio= 0.1

#create_image_class(image_dir, height, width, name_of_class, output_path,test_split_ratio= 0.1)

image_dir = 'manuel_sorted/boobs_search/1'

name_of_class = "1/"

create_image_class(image_dir, height, width, name_of_class, output_path,test_split_ratio= 0.1)