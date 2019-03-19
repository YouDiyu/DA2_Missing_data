# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 17:24:05 2018

@author: You
"""

import numpy as np
from keras.models import Sequential
from keras import layers
from keras.utils.generic_utils import to_list
from keras import backend as K

def floatornone(x):
    try:
        x=float(x)
    except ValueError:
        x=None
    return x
def generate_mask(train_data):
    mask=np.minimum(train_data,0)
    mask=np.maximum(mask,1)
    mask[np.isnan(mask)]=0
    return mask
def mask_weighting(mask):
    maskweights=np.sum(mask,axis=-1)
    maskweights_matrix=(mask.T/maskweights*mask.shape[-1]).T
    return maskweights_matrix
#zero-mean-normalization
def data_preprocess_activedrop(train_data):
    
    mask_binary=generate_mask(train_data)
    #nan replace with zero
    train_data_zero=train_data.copy()
    train_data_zero[np.isnan(train_data)]=0
    train_mean=train_data_zero.mean(axis=0)
    mean=train_mean*train_data.shape[0]/np.sum(mask_binary,axis=0)
    #nan replace with mean
    train_data_mean=train_data.copy()
    for i in range(train_data.shape[-1]):
        train_data_mean[:,i][np.isnan(train_data[:,i])]=mean[i]
    train_std=train_data_mean.std(axis=0)
    std=train_std*train_data.shape[0]/np.sum(mask_binary,axis=0)
    #processed data
    processed_data=(train_data_mean-mean)/std
    processed_data=processed_data*mask_weighting(mask_binary)
    return processed_data
fname='E:/Mashine lerning/mammographic-masses-DA/mammographic_masses.txt'
f=open(fname)
data=f.read()
f.close()
lines=data.split('\n')
float_data=np.zeros((len(lines),6))

for i,line in enumerate(lines):
    values=np.zeros(6)
    for j,x in enumerate(line.split(',')[:]):
        x=floatornone(x)
        values[j]=x
    float_data[i,:]=values



