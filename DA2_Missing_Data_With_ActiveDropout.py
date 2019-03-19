# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:37:39 2018

@author: You
"""

import numpy as np
from keras.models import Sequential
from keras import layers
import You
from keras import regularizers
from keras import backend as K

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
    #40% Drop
    
'''    
train_data=float_data[:,:-1]
train_data=K.dropout(train_data,0.4,None,None)
train_data=K.get_value(train_data)
train_data=dp.zero_mean_normalization(train_data)
train_data*=0.6
label_data=float_data[:,-1] 
    
'''
train_data=float_data[:,:-1]
train_data=dp.zero_mean_normalization(train_data)
#train_data=K.dropout(train_data,0.4,None,None)
#train_data=K.get_value(train_data)
#train_data*=0.6
#train_data=dp.zero_mean_normalization(train_data)
label_data=float_data[:,-1] 

'''
train_data=float_data[:,:-1]
train_data=dp.zero_mean_normalization(train_data)

label_data=float_data[:,-1]
'''

'''
train_data=dp.none_to_zero(train_data)
#data preprocess
mean=train_data.mean(axis=0)
train_data-=mean
std=train_data.std(axis=0)
train_data/=std
'''
model = Sequential()
#model.add(You.activeDrop())
#model.add(layers.Dense(8,activation='relu',kernel_regularizer=regularizers.l2(0.1)))
#model.add(layers.RNN(32,))
model.add(layers.Dense(8,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
# Compile model
model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
# Fit
history = model.fit(train_data, label_data,epochs=100,batch_size=64,validation_split=0.2)
# Plot
'''
#model2
model1 = Sequential()
#model.add(You.activeDrop())
model1.add(layers.Dense(2,activation='relu'))
#model.add(layers.RNN(32,))
model1.add(layers.Dense(4,activation='relu'))
model1.add(layers.Dense(1,activation='sigmoid'))
# Compile model
model1.compile(optimizer='Adam',loss='mse',metrics=['acc'])
# Fit
history1 = model1.fit(train_data, label_data,epochs=100,batch_size=64,validation_split=0.2)
'''

import matplotlib.pyplot as plt
acc = history.history['acc']
#acc1 = history1.history['acc']
val_acc = history.history['val_acc']
#val_acc1 = history1.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'bo', label='Validation acc')
plt.xlabel('Epoche')
plt.ylabel('Value')
#plt.plot(epochs, acc1[39:100], 'r', label='Training acc 1')
#plt.plot(epochs, val_acc1, 'r', label='Validation acc 1')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

model.summary()
w=model.layers[0].get_weights()[0]
mean_w=w.mean(axis=0)
std_w=w.std(axis=0)
abs_w=abs(mean_w)
plt.title('mean and std of weights')
plt.xlabel('Neuron Nr.')
plt.ylabel('Value')
plt.plot(range(1,len(abs_w)+1),abs_w,'bo',label='weights_mean')
plt.plot(range(1,len(std_w)+1),std_w,'ro',label='weights_std')
plt.legend()
plt.show()
def get_activity(model,inputs):
    outputs=K.function([model.input],[model.layers[1].output])
    return outputs
outputs=K.function([model.input],[model.layers[1].output])