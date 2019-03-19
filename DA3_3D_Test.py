

import numpy as np
from keras.models import Sequential
from keras import layers
from keras import regularizers
import keras.backend as K
import You
import pandas as pd
from sklearn.preprocessing import Imputer
from fancyimpute import IterativeImputer
dp=You.data_preprocess()
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

fname='E:/Mashine lerning/Breast Cancer Wisconsin/Breast Cancer Wisconsin.txt'
f=open(fname)
data=f.read()
f.close()
lines=data.split('\n')
float_data=np.zeros((len(lines),32))

for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[:]]
    float_data[i,:]=values


# Train Set
float_data=shuffel(float_data)
#Nrmalisation train_data
float_data[:,2:]=dp.zero_mean_normalization(float_data[:,2:])
data=float_data[:,2:]
data=np.expand_dims(data,axis=-1)
label_data=float_data[:,1]

model = Sequential()
model.add(You.GRU_M(16,return_sequences=True))
model.add(You.AttentionLayer_3D(supmask=True))
#model.add(You.activeDrop())
#model.add(layers.Dense(16,activation='relu'))
#model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
model.compile(optimizer='Adam',loss='mse',metrics=['acc'])
history = model.fit(data, label_data,validation_split=0.2,epochs=150,batch_size=64)

import matplotlib.pyplot as plt


acc1 = history.history['acc']
val_acc1 = history.history['val_acc']
epochs = range(1, len(acc1) + 1)
plt.plot(epochs, acc1, 'b', label='Training acc voll')    
plt.plot(epochs, val_acc1, 'r', label='Validation acc voll')    
plt.xlabel('Epochs')
plt.ylabel('Accuracy')

plt.legend()
plt.show()




