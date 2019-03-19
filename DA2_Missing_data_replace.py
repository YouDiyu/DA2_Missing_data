# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 19:13:41 2018

@author: You
"""


import numpy as np
from keras.models import Sequential
from keras import layers
import You
from keras import regularizers
import pandas as pd
dp=You.data_preprocess()
#load data
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
    
float_data=pd.DataFrame(float_data)
float_data=float_data.dropna()
float_data=np.array(float_data)
        
train_data=float_data[:,:-1]
label_data=float_data[:,-1]

train_data[np.isnan(train_data)]=0
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std

model = Sequential()
model.add(layers.Dropout(0.4))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Dense(4,activation='sigmoid'))
model.add(layers.Dense(1))
# Compile model
model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
# Fit
history = model.fit(train_data, label_data,epochs=40,batch_size=32,validation_split=0.2)
# Plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

model.summary()
