# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 23:30:44 2019

@author: You
"""


import numpy as np
from keras.models import Sequential
from keras import layers
from keras import regularizers
import keras.backend as K
import You
import pandas as pd
from sklearn.preprocessing import Imputer
from fancyimpute import IterativeImputer
from keras import initializers
def multi_imp(data,m):
    XY=data
    n_imputations = m
    XY_completed = []
    for i in range(n_imputations):
        imputer = IterativeImputer(n_iter=5, sample_posterior=True, random_state=i)
        XY_completed.extend(imputer.fit_transform(XY))
    return np.array(XY_completed)
def to_pandas(data):
    data=pd.DataFrame(data)
    return data
def elim(data):
    elim=data.dropna()
    return elim
def together(train_data,label_data):
    data=np.zeros((len(train_data),len(train_data[0])+1))
    data[:,:len(train_data[0])]=train_data
    data[:,len(train_data[0])]=label_data
    return data
def zerotonan(train_data):
    train_data=(train_data/train_data)*train_data
    return train_data
def verarbeit_elim(train_data,label_data):
    train_data=zerotonan(train_data)
    data=together(train_data,label_data)
    dropd=elim(to_pandas(data))
    train_data=np.array(dropd)[:,:-1]
    label_data=np.array(dropd)[:,-1]
    return train_data,label_data

def shuffel(data):
    data=pd.DataFrame(data)
    data.sample(frac=1.0)
    data=np.array(data)
    return data
def verarbeit_meanimp(train_data,label_data):
    train_data=zerotonan(train_data)
    data=together(train_data,label_data)
    imp=Imputer(missing_values=np.nan , strategy='mean', axis=0)
    imp.fit(data)
    data=imp.transform(data)
    train_data=data[:,:-1]
    label_data=data[:,-1]
    return train_data,label_data
def verarbeit_mulimp(train_data,label_data):
    train_data=zerotonan(train_data)
    data=together(train_data,label_data)
    data_mi=multi_imp(data,5)
    data_mi=shuffel(data_mi)
    train_data=data_mi[:,:-1]
    label_data=data_mi[:,-1]
    return train_data,label_data
dp=You.data_preprocess()
fname='E:/Mashine lerning/mammographic-masses-DA/mammographic_masses.txt'
f=open(fname)
data=f.read()
f.close()
lines=data.split('\n')
float_data=np.zeros((len(lines),6))

for i,line in enumerate(lines):
    values=np.zeros(6)
    #fehlende Daten ersetzen mit None
    for j,x in enumerate(line.split(',')[:]):
        x=dp.floatornone(x)
        values[j]=x
    float_data[i,:]=values

# Train Set
float_data=shuffel(float_data)

#Nrmalisation train_data
float_data[:,1:-1]=dp.zero_mean_normalization(float_data[:,1:-1])
data=float_data[:,1:-1]


train_data=data[150:]
label_data=float_data[150:,-1]

#validation data
# MCAR
train_data_voll=K.dropout(train_data,0.0001,None,None)
train_data_voll=K.get_value(train_data_voll)

train_data_20_MCAR=K.dropout(train_data,0.2,None,None)
train_data_20_MCAR=K.get_value(train_data_20_MCAR)
train_data_20_MCAR*=0.8

train_data_40_MCAR=K.dropout(train_data,0.4,None,None)
train_data_40_MCAR=K.get_value(train_data_40_MCAR)
train_data_40_MCAR*=0.6


#MAR
sort_float_data=sorted(float_data[150:], key=lambda x: x[1])
sort_float_data=np.array(sort_float_data)
sort_train_data=sort_float_data[:,1:-1]
label_data_MAR=sort_float_data[:,-1]

data_20_MAR=train_data
a=K.dropout(sort_train_data[:400,:],0.15)
data_20_MAR[:400,:]=K.get_value(a)
b=a=K.dropout(sort_train_data[400:,:],0.25)
data_20_MAR[400:,:]=K.get_value(b)
data_20_MAR=together(data_20_MAR,label_data_MAR)
data_20_MAR=shuffel(data_20_MAR)
train_data_20_MAR=data_20_MAR[:,:-1]
label_data_20_MAR=data_20_MAR[:,-1]

data_40_MAR=train_data
a=K.dropout(sort_train_data[:400,:],0.3)
data_40_MAR[:400,:]=K.get_value(a)
b=a=K.dropout(sort_train_data[400:,:],0.5)
data_40_MAR[400:,:]=K.get_value(b)
data_40_MAR=together(data_40_MAR,label_data_MAR)
data_40_MAR=shuffel(data_40_MAR)
train_data_40_MAR=data_40_MAR[:,:-1]
label_data_40_MAR=data_40_MAR[:,-1]



train_data_voll_elim,label_data_voll_elim=zerotonan(train_data_voll),label_data
train_data_20_MCAR_elim,label_data_20_MCAR_elim=zerotonan(train_data_20_MCAR),label_data
train_data_40_MCAR_elim,labe_data_40_MCAR_elim=zerotonan(train_data_40_MCAR),label_data

train_data_20_MAR_elim,label_data_20_MAR_elim=zerotonan(train_data_20_MAR),label_data_20_MAR
train_data_40_MAR_elim,label_data_40_MAR_elim=zerotonan(train_data_40_MAR),label_data_40_MAR


def ADmodel(train_data,label_data,val_data1,val_data2):
    model = Sequential()
    model.add(You.activeDrop())
    model.add(layers.Dense(8,activation='relu', kernel_initializer=initializers.random_normal(seed=1),bias_initializer=initializers.random_normal(seed=2)))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid', kernel_initializer=initializers.random_normal(seed=3),bias_initializer=initializers.random_normal(seed=4)))
    model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
    history = model.fit(train_data, label_data,validation_data=(val_data1,val_data2),epochs=50,batch_size=32)
    adloss = history.history['loss']
    return adloss
#'validation_data=(val_data1,val_data2)'validation_split=0.2
adloss_voll=ADmodel(train_data_voll_elim,label_data_voll_elim,data[:150,],float_data[:150,-1])

adloss_20_MCAR=ADmodel(train_data_20_MCAR_elim,label_data_20_MCAR_elim,data[:150,],float_data[:150,-1])
adloss_40_MCAR=ADmodel(train_data_40_MCAR_elim,labe_data_40_MCAR_elim,data[:150,],float_data[:150,-1]) 

adloss_20_MAR=ADmodel(train_data_20_MAR_elim,label_data_20_MAR_elim,data[:150,],float_data[:150,-1])
adloss_40_MAR=ADmodel(train_data_40_MAR_elim,label_data_40_MAR_elim,data[:150,],float_data[:150,-1]) 

train_data_voll_elim,label_data_voll_elim=verarbeit_meanimp(train_data_voll,label_data)
train_data_20_MCAR_elim,label_data_20_MCAR_elim=verarbeit_meanimp(train_data_20_MCAR,label_data)
train_data_40_MCAR_elim,labe_data_40_MCAR_elim=verarbeit_meanimp(train_data_40_MCAR,label_data)

train_data_20_MAR_elim,label_data_20_MAR_elim=verarbeit_meanimp(train_data_20_MAR,label_data_20_MAR)
train_data_40_MAR_elim,label_data_40_MAR_elim=verarbeit_meanimp(train_data_40_MAR,label_data_40_MAR)


def MImodel(train_data,label_data,val_data1,val_data2):
    model = Sequential()
    model.add(layers.Dense(8,activation='relu', kernel_initializer=initializers.random_normal(seed=1),bias_initializer=initializers.random_normal(seed=2)))
    #model.add(layers.Dense(4,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid', kernel_initializer=initializers.random_normal(seed=3),bias_initializer=initializers.random_normal(seed=4)))
    model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
    history = model.fit(train_data, label_data,validation_data=(val_data1,val_data2),epochs=50,batch_size=32)
    miloss = history.history['loss']
    return miloss
#'validation_data=(val_data1,val_data2)'validation_split=0.2
miloss_voll=MImodel(train_data_voll_elim,label_data_voll_elim,data[:150,],float_data[:150,-1])

miloss_20_MCAR=MImodel(train_data_20_MCAR_elim,label_data_20_MCAR_elim,data[:150,],float_data[:150,-1])
miloss_40_MCAR=MImodel(train_data_40_MCAR_elim,labe_data_40_MCAR_elim,data[:150,],float_data[:150,-1]) 

miloss_20_MAR=MImodel(train_data_20_MAR_elim,label_data_20_MAR_elim,data[:150,],float_data[:150,-1])
miloss_40_MAR=MImodel(train_data_40_MAR_elim,label_data_40_MAR_elim,data[:150,],float_data[:150,-1]) 


import matplotlib.pyplot as plt

epochs = range(1, len(adloss_voll) + 1)
plt.plot(epochs, adloss_voll, 'b', label='AD loss voll')
plt.plot(epochs, adloss_20_MCAR, 'r', label='AD loss 20 MCAR')
plt.plot(epochs, adloss_40_MCAR, 'g', label='AD loss 40 MCAR')
plt.plot(epochs, adloss_20_MAR, 'c', label='AD loss 20 MAR')
plt.plot(epochs, adloss_40_MAR, 'm', label='AD loss 40 MAR')
plt.plot(epochs, miloss_voll, 'b.', label='MI loss voll')
plt.plot(epochs, miloss_20_MCAR, 'r.', label='MI loss 20 MCAR')
plt.plot(epochs, miloss_40_MCAR, 'g.', label='MI loss 40 MCAR')
plt.plot(epochs, miloss_20_MAR, 'c.', label='MI loss 20 MAR')
plt.plot(epochs, miloss_40_MAR, 'm.', label='MI loss 40 MAR')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Fehlerfunktion')
plt.legend()
plt.show()
