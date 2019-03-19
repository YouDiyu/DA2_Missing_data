# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 18:02:47 2019

@author: You
"""


import numpy as np
from keras.models import Sequential
from keras import layers
from keras import regularizers
import keras.backend as K
import You
import pandas as pd
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
float_data=pd.DataFrame(float_data)
float_data.sample(frac=1.0)
float_data=np.array(float_data)

train_data=float_data

data=dp.zero_mean_normalization(float_data[:,1:-1])


train_data=data[150:]
label_data=float_data[150:,-1]

train_data_voll=train_data#validation data


train_data_20=K.dropout(train_data,0.2,None,None)
train_data_20=K.get_value(train_data_20)
train_data_20*=0.8

train_data_40=K.dropout(train_data,0.4,None,None)
train_data_40=K.get_value(train_data_40)
train_data_40*=0.6

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

train_data_voll,label_data_voll=verarbeit_elim(train_data_voll,label_data)
train_data_20,label_data_20=verarbeit_elim(train_data_20,label_data)
train_data_40,label_data_40=verarbeit_elim(train_data_40,label_data)



def NNmodel(train_data,label_data,val_data1,val_data2):
    model = Sequential()
    model.add(layers.Dense(8,activation='relu'))
    model.add(layers.Dense(1,activation='sigmoid'))
    model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
    history = model.fit(train_data, label_data,validation_data=(val_data1,val_data2),epochs=50,batch_size=64)
    acc1 = history.history['acc']
    val_acc1 = history.history['val_acc']
    return acc1,val_acc1
#'validation_data=(val_data1,val_data2)'validation_split=0.2
acc_voll, val_acc_voll=NNmodel(train_data_voll,label_data_voll,data[:150,],float_data[:150,-1])
acc_20, val_acc_20=NNmodel(train_data_20,label_data_20,data[:150,],float_data[:150,-1])
acc_40, val_acc_40=NNmodel(train_data_40,label_data_40,data[:150,],float_data[:150,-1]) 

import matplotlib.pyplot as plt

epochs = range(1, len(acc_voll) + 1)
plt.plot(epochs, acc_voll, 'bo', label='Training acc voll')
plt.plot(epochs, acc_20, 'ro', label='Training acc 20')
plt.plot(epochs, acc_40, 'go', label='Training acc 40')
plt.plot(epochs, val_acc_voll, 'b', label='Validation acc voll')
plt.plot(epochs, val_acc_20, 'r', label='Validation acc 20')
plt.plot(epochs, val_acc_40, 'g', label='Validation acc 40')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()