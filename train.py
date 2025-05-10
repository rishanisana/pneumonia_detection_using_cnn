# -*- coding: utf-8 -*-
"""
Created on Sat May  2 16:19:14 2020

@author: Anuj
"""

# Making the imports
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
import os

# Code for loading training and validation data at the time of training
base_dir = os.getcwd()  # Getting current directory

target_shape = (224, 224)  # Defining the input shape
train_dir = os.path.join(base_dir, "chest_xray/train")  # Fixed path
val_dir = os.path.join(base_dir, "chest_xray/val")  # Directories for data
test_dir = os.path.join(base_dir, "chest_xray/test")  #

# Loading the VGG16 model with ImageNet weights without the FC layers
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in vgg.layers:
    layer.trainable = False  # Making all the layers non-trainable

x = Flatten()(vgg.output)  # Flattening out the last layer
predictions = Dense(2, activation='softmax')(x)  # Dense layer for pneumonia prediction
model = Model(inputs=vgg.input, outputs=predictions)
model.summary()

# Data augmentation for training
train_gen = ImageDataGenerator(rescale=1/255.0,
                               horizontal_flip=True,
                               zoom_range=0.2,
                               shear_range=0.2)  # Data loader for training

test_gen = ImageDataGenerator(rescale=1/255.0)  # Data loader for validation/testing

# Creating data generators
train_data_gen = train_gen.flow_from_directory(train_dir,
                                               target_size=target_shape,
                                               batch_size=16,
                                               class_mode='categorical')

test_data_gen = test_gen.flow_from_directory(test_dir,  # Used `test_gen` instead of `train_gen`
                                             target_size=target_shape,
                                             batch_size=16,
                                             class_mode='categorical')

plot_model(model, to_file='model.png')

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model
hist = model.fit(train_data_gen,
                 steps_per_epoch=10,
                 epochs=10,
                 validation_data=test_data_gen,
                 validation_steps=10)

# Plotting training history
plt.style.use("ggplot")
plt.figure()
plt.plot(hist.history["loss"], label="train_loss")
plt.plot(hist.history["val_loss"], label="val_loss")
plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Training")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig("epochs.png")

train_acc = hist.history['accuracy'][-1]
val_acc = hist.history['val_accuracy'][-1]

print(f"Final Training Accuracy: {train_acc*100:.2f}%")
print(f"Final Validation Accuracy: {val_acc*100:.2f}%")