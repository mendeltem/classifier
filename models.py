#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:03:41 2019

@author: pandoora
"""
import tensorflow as tf
keras = tf.keras
models = keras.models

load_model = keras.models.load_model
optimizers = keras.optimizers
Input = keras.layers.Input
Dense = keras.layers.Dense
Flatten = keras.layers.Flatten
Activation = keras.layers.Activation
Conv2D = keras.layers.Conv2D
Conv2DTranspose = keras.layers.Conv2DTranspose
MaxPooling2D  = keras.layers.MaxPooling2D
Dropout = keras.layers.Dropout
Model = keras.models.Model
ImageDataGenerator= keras.preprocessing.image.ImageDataGenerator
concatenate = keras.layers.concatenate
ModelCheckpoint = keras.callbacks.ModelCheckpoint
VGG19 = keras.applications.VGG19
BatchNormalization = keras.layers.BatchNormalization


def classifier(img_width, img_height, num_class):

    input_shape = (img_width,img_height, 3)
    vgg= VGG19(include_top=False,input_shape=input_shape,
        weights='imagenet')
    vgg.trainable = False
    
    model = models.Sequential()
    
    model.add(vgg)
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))    
    model.add(Dense(num_class, activation='softmax'))
    
    
    model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
    
    return model