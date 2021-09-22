# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 13:42:25 2021

@author: ahmed
"""


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import numpy as np

import os
import gc
import random
from tqdm import tqdm

import cv2
import PIL.Image
import PIL.ImageOps
from IPython.display import Image, display
from patchify import patchify, unpatchify

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D, Dropout, Activation, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, EarlyStopping
from tensorflow.keras.metrics import MeanIoU

import segmentation_models as sm


patch_size = 256
def format_img(img):
    
    size_x = (img.shape[1] // patch_size) * patch_size
    size_y = (img.shape[0] // patch_size) * patch_size
    
    img = PIL.Image.fromarray(img)
    img = img.crop((0, 0, size_x, size_y))
    
    img = np.array(img)
    
    return img


def create_single_img_patch(img_patches, dataset):
    
    for i in range(img_patches.shape[0]):
        for j in range(img_patches.shape[1]):
            
            single_img_patch = img_patches[i,j,:,:]
            
            # Scaling the img patch so that the values ranges from 0 to 1
            # single_img_patch = (single_img_patch.astype('float32')) / 255.0
            
            # Drop the extra dimension that patchify adds
            single_img_patch = single_img_patch[0]
            
            dataset.append(single_img_patch)




# Creating Images DataSet
dataset_path = "E:/Software/professional practice projects/In Progress 2/Semantic segmentation dataset"
images_dataset = []


for tile in os.listdir(dataset_path):
    
    if not tile.endswith('.json'):
        
        tile_path = os.path.join(dataset_path, tile)
        tile_images_path = os.path.join(tile_path, 'images')
        
        for img_name in os.listdir(tile_images_path):
            
            if img_name.endswith(".jpg"):
                
                img_path = os.path.join(tile_images_path, img_name)
                img = cv2.imread(img_path, 1)
                # img = Image.open(img_path)
                
                img = format_img(img)
                img = (img / 255.0)
                
                # extract patches from each image
                # step = 256 for patches with size 256 means no overlap
                img_pathes = patchify(img, (patch_size, patch_size, 3), step=patch_size)
                
                
                create_single_img_patch(img_pathes, images_dataset)




# Creating Masks DataSet
masks_dataset = []

for tile in os.listdir(dataset_path):
    
    if not tile.endswith('.json'):
        
        tile_path = os.path.join(dataset_path, tile)
        tile_masks_path = os.path.join(tile_path, 'masks')
    
        for mask_name in os.listdir(tile_masks_path):
            
            if mask_name.endswith(".png"):
                mask_path = os.path.join(tile_masks_path, mask_name)
                mask = cv2.imread(mask_path)
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        
                mask = format_img(mask)
        
                mask_patches = patchify(mask, (patch_size, patch_size, 3), step=patch_size)
        
                create_single_img_patch(mask_patches, masks_dataset)


# Converting from lists to arrays
images_dataset = np.array(images_dataset)
masks_dataset = np.array(masks_dataset)



# Visualizing random images with its masks

plt.figure(figsize=(10, 10))
idx = 0
for i in range(2):
    
    idx += 1
    plt.subplot(2, 2, idx)
    
    rand_idx = random.choice(range(0, len(images_dataset)))
    
    img = images_dataset[rand_idx]
    plt.imshow(img)
    plt.title("Original")
    
    idx += 1
    plt.subplot(2, 2, idx)
    mask = masks_dataset[rand_idx]
    plt.imshow(mask)
    plt.title('Mask')

plt.show()

# these HEX are provided by kaggle dataset 
"""
Building: #3C1098
Land (unpaved area): #8429F6
Road: #6EC1E4
Vegetation: #FEDD3A
Water: #E2A929
Unlabeled: #9B9B9B

"""

Building = '3C1098'
Land = '8429F6'
Road = '6EC1E4'
Vegetation = 'FEDD3A'
Water = 'E2A929'
Unlabeled = '9B9B9B'


# Coverting these HEX (base 16) into RGB array

Building = np.array(tuple(int(Building[i:i+2], 16) for i in (0, 2, 4)))

Land = np.array(tuple(int(Land[i:i+2], 16) for i in (0, 2, 4)))

Road = np.array(tuple(int(Road[i:i+2], 16) for i in (0, 2, 4)))

Vegetation = np.array(tuple(int(Vegetation[i:i+2], 16) for i in (0, 2, 4)))

Water = np.array(tuple(int(Water[i:i+2], 16) for i in (0, 2, 4)))

Unlabeled = np.array(tuple(int(Unlabeled[i:i+2], 16) for i in (0, 2, 4)))





def rgb_to_2D_label(label):
    
    seg_label = np.zeros(label.shape, dtype=np.uint8)
    
    # find all pixels of the label that matches the RGB arrays above (e.g. building, land, etc..)
    # if matches we replace all pixels' values to specific integer (class)
    seg_label [np.all(label == Building, axis=-1)] = 0
    seg_label [np.all(label == Land, axis=-1)] = 1
    seg_label [np.all(label == Road, axis=-1)] = 2
    seg_label [np.all(label == Vegetation, axis=-1)] = 3
    seg_label [np.all(label == Water, axis=-1)] = 4
    seg_label [np.all(label == Unlabeled, axis=-1)] = 5
    
    seg_label = seg_label[:,:,0] # no need for all channels
    
    return seg_label
    

# testing the rgb_to_2D_label function
label = masks_dataset[5]
label2 = rgb_to_2D_label(label)

# Creating labels list
labels = []

for i in range(masks_dataset.shape[0]):
    label = rgb_to_2D_label(masks_dataset[i])
    labels.append(label)


labels = np.array(labels)

# make the labels ready to be fed into deep nueral network (e.g. UNet,..)
labels = np.expand_dims(labels, axis=3)


print("unique labels are: ", np.unique(labels))



# checking that every thing is ok after thses conversions and ops

plt.figure(figsize=(10, 10))
idx = 0
for i in range(2):
    
    idx += 1
    plt.subplot(2, 2, idx)
    
    rand_idx = random.choice(range(0, len(images_dataset)))
    
    img = images_dataset[rand_idx]
    plt.imshow(img)
    plt.title("Original")
    
    idx += 1
    plt.subplot(2, 2, idx)
    mask = labels[rand_idx][:,:,0]
    plt.imshow(mask)
    plt.title('Mask')

plt.show()

num_classes = len(np.unique(labels))

# one hot encodeing labels
labels = to_categorical(labels)


# split data
x_train, x_test, y_train, y_test = train_test_split(images_dataset, labels, test_size=0.2)

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)


################ Creating UNet Model ##########################
""" Encoder """



def conv2D_block(input_tensor, n_filters, kernel_size=3):
    
    x = input_tensor
    
    for i in range(2):
    
        x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size), padding='same')(x)
        
        x = Activation('relu')(x)
    
    return x




def encoder_block(inputs, n_filters, pool_size, dropout):
    
    conv_output = conv2D_block(inputs, n_filters = n_filters)
    
    x = MaxPooling2D(pool_size=pool_size)(conv_output)
    
    x = Dropout(dropout)(x)
    
    return conv_output, x


def Encoder(inputs):
    
    conv_out_1, x1 = encoder_block(inputs, n_filters=64, pool_size=(2, 2), dropout=0.3)
    
    conv_out_2, x2 = encoder_block(x1, n_filters=128, pool_size=(2, 2), dropout=0.3)
    
    conv_out_3, x3 = encoder_block(x2, n_filters=128, pool_size=(2, 2), dropout=0.3)
    
    conv_out_4, x4 = encoder_block(x3, n_filters=512, pool_size=(2, 2), dropout=0.3)
    
    return x4, (conv_out_1, conv_out_2, conv_out_3, conv_out_4)



def Bottleneck(inputs):
    
    bottle_neck = conv2D_block(inputs, n_filters=1024)
    
    return bottle_neck




""" Decoder"""

def decoder_block(inputs, conv_output, n_filters, kernel_size, strides, dropout):
    
    trans = Conv2DTranspose(n_filters, kernel_size, strides=strides, padding='same')(inputs)
    
    conct = concatenate([trans, conv_output])
    
    x = Dropout(dropout)(conct)
    
    x = conv2D_block(x, n_filters, kernel_size=3)
    
    return x




def Decoder(inputs, convs, num_classes):
    
    conv_1, conv_2, conv_3, conv_4 = convs
    
    x1 = decoder_block(inputs, conv_4, n_filters=512, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    x2 = decoder_block(x1, conv_3, n_filters=256, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    x3 = decoder_block(x2, conv_2, n_filters=128, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    x4 = decoder_block(x3, conv_1, n_filters=64, kernel_size=(3, 3), strides=(2, 2), dropout=0.3)
    
    outputs = Conv2D(num_classes, kernel_size=(1, 1), activation='softmax', padding='same')(x4)
    
    return outputs





def UNet(num_classes):
    
    inputs = Input(shape=(256, 256, 3))
    
    encoder_output, convs = Encoder(inputs)
    
    bottle_neck = Bottleneck(encoder_output)
    
    outputs = Decoder(bottle_neck, convs, num_classes)
    
    model = Model(inputs=inputs, outputs=outputs)
    
    return model


model = UNet(num_classes)

model.summary()

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])


my_callbacks = [ModelCheckpoint(filepath="model.h5", monitor='val_loss', verbose=1, save_best_only=True),
                EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
                CSVLogger("train_performance_per_epoch.csv")]

history = model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=1, 
                    callbacks=my_callbacks, validation_data=(x_val, y_val))




model.save('Aerial_Imagery_Model.h5')
model.save_weights('Aerial_Imagery_Weights.h5')

json_model = model.to_json()

with open("E:/Software/professional practice projects/In Progress 2/Aerial_Imagery_Model.json", 'w') as json_file:
    json_file.write(json_model)



# Plot Accuracy
# I used model.history.history because i stopped model training so history varibale haven't been recorded in the kernel
plt.figure(figsize=(8, 8))
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])
plt.ylabel("accuracy")
plt.xlabel("Epochs")
plt.title("Aerial Imagery Model Accuracy")
plt.legend(['accuracy', 'val_accuracy'], loc='lower right')


# Plot Loss
plt.figure(figsize=(8, 6))
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])
plt.xlabel('Epochs')
plt.ylabel("Loss")
plt.title("Aerial Imagery Model Loss")
plt.legend(['loss', 'val_loss'])




# Making Predictions
y_pred = model.predict(x_test)
y_pred_argmx = np.argmax(y_pred, axis=3)

# Ground Truth armax
y_test_argmx = np.argmax(y_test, axis=3)


# Calculate IOU metric

M_IoU = MeanIoU(num_classes=num_classes)
M_IoU.update_state(y_true=y_test_argmx, y_pred=y_pred_argmx)

print("Mean IoU = ", M_IoU.result().numpy())




# Exploring predictions on random images from test dataset
plt.figure(figsize=(14, 14))
cnt = 0
plt.suptitle("Exploring predictions on random images from test dataset", fontweight="bold", fontsize='xx-large')

for i in range(3):
    
    rand_idx = random.randint(0, x_test.shape[0])
    
    cnt += 1
    plt.subplot(3, 3, cnt)
    plt.imshow(x_test[rand_idx])
    plt.title("Original")
    
    cnt += 1
    plt.subplot(3, 3, cnt)
    plt.imshow(y_test_argmx[rand_idx])
    plt.title("True Mask")
    
    cnt += 1
    plt.subplot(3, 3, cnt)
    plt.imshow(y_pred_argmx[rand_idx])
    plt.title("Pred Mask")












