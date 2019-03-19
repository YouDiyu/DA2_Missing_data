# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 15:34:15 2018

@author: You
"""

import numpy as np
from keras.models import Sequential
from keras import layers
from keras import regularizers
import keras.backend as K
import You
import pandas as pd
def shuffel(data):
    data=pd.DataFrame(data)
    data.sample(frac=1.0)
    data=np.array(data)
    return data
dp=You.data_preprocess()
fname='E:/Mashine lerning/Diabetic Retinopathy Debrecen-DA/DRDataset.txt'
f=open(fname)
data=f.read()
f.close()
lines=data.split('\n')
float_data=np.zeros((len(lines),20))

for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[:]]
    float_data[i,:]=values

# Train Set
float_data=shuffel(float_data)
    
train_data=float_data[:,:-1]
label_data=float_data[:,-1]
train_data=train_data/train_data*train_data
train_data=dp.zero_mean_normalization(train_data)

#train_data=np.expand_dims(train_data,axis=-1)
#train_data=dp.zero_mean_normalization(train_data)


# Label 


# creat model
model = Sequential()
model.add(You.activeDrop(0.3))
model.add(layers.Dense(16,activation='relu',kernel_regularizer=regularizers.l2(0.00)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
# Compile model
model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
# Fit
history = model.fit(train_data, label_data,epochs=100,batch_size=64,validation_split=0.2)
# Plot
import matplotlib.pyplot as plt
acc1 = history.history['acc']
val_acc1 = history.history['val_acc']
loss1 = history.history['loss']
val_loss1 = history.history['val_loss']
epochs = range(1, len(acc1) + 1)
plt.plot(epochs, acc1, 'bo', label='Training acc')
plt.plot(epochs, val_acc1, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()
##############################################################2
'''
train_data2=float_data[:430,:19]
label_data2=float_data[:430,-1]

model2 = Sequential()
model2.add(layers.Dense(8,activation='relu',kernel_regularizer=regularizers.l2(0.05)))
model2.add(layers.Dense(4,activation='relu'))
model2.add(layers.Dense(1,activation='sigmoid'))
# Compile model
model2.compile(optimizer='Adam',loss='mse',metrics=['acc'])
# Fit
history = model2.fit(train_data2, label_data2,epochs=100,batch_size=64,validation_split=0.2)

import matplotlib.pyplot as plt
acc2 = history.history['acc']
val_acc2 = history.history['val_acc']
loss2 = history.history['loss']
val_loss2 = history.history['val_loss']
epochs = range(1, len(acc2) + 1)
plt.plot(epochs, acc2, 'bo', label='Training acc')
plt.plot(epochs, val_acc2, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

import matplotlib.pyplot as plt

epochs = range(1, len(acc2) + 1)
plt.plot(epochs, val_acc1, 'r', label='vollständiger Datensatz')
plt.plot(epochs, val_acc2, 'g', label='40% Datensatz')
plt.xlabel('Epoche')
plt.ylabel('Genauigkeit')
plt.title('Validierungsgenauigkeit')
plt.legend()
plt.show()
'''