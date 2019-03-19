# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:12:39 2018

@author: You
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 18:37:39 2018

@author: You
"""

import numpy as np
from keras.models import Sequential
from keras import layers
import You
import tensorflow as tf
from keras import regularizers
import keras.backend as K
sess=tf.Session()

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

train_data=float_data[:,1:-1]
train_data=dp.zero_mean_normalization(train_data)
train_data=K.dropout(train_data,0.2,None,None)
train_data=K.get_value(train_data)
train_data*=0.8
train_data=np.expand_dims(train_data,axis=-1)
#train_data=dp.zero_mean_normalization(train_data)
label_data=float_data[:,-1] 


#train_data=np.load('train_data_mamm_0.4.npy')
#label_data=np.load('train_label_mamm_0.4.npy')

'''
model.add(You.activeDrop(0.4))
'''

model = Sequential()
#model.add(layers.Embedding(100, 128,input_length=5))
model.add(You.GRU_M(8,return_sequences=True,input_shape=(4,1)))
#model.add(layers.GRU(8,return_sequences=True,return_state=True,input_shape=(5,1)))
model.add(You.AttentionLayer_3D(supmask=True))
#model.add(layers.Flatten())
#model.add(layers.Dense(5,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))
# Compile model
model.compile(optimizer='rmsprop',
loss='binary_crossentropy',
metrics=['acc'])
# Fit
history = model.fit(train_data, label_data,epochs=100,batch_size=32,validation_split=0.2)
# Plot
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

model.summary()

def get_att(model,inputs):
    att_b_w=model.layers[1].get_weights()
    att_w=tf.convert_to_tensor(att_b_w[0])
    att_b=tf.convert_to_tensor(att_b_w[1])
    outputs=K.function([model.input],[model.layers[0].output])
    outputs_at_0=np.array(outputs([inputs]))
    x=K.permute_dimensions(outputs_at_0,(0,1,3,2))
    z=K.dot(x,att_w)+att_b
    activation=K.tanh(z)
    expatt=K.exp(activation)
    att=expatt/K.cast(K.sum(expatt,axis=3,keepdims=True)+K.epsilon(),K.floatx())
    att_sum=K.get_value(K.sum(att,axis=2))
    return att_sum
#K.get_value(K.sum(aaa,axis=2))
 
def get_attwerte(model,inputs):
    outputs=K.function([model.input],[model.layers[1].att])
    return outputs
    

import matplotlib.pyplot as plt
def plotshow(arr):
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    #10
    xs=range(1,(len(arr)+1))
    #5
    for i in xs:
        ys=range(1,(len(arr[0])+1))
        z=arr[(i-1),:]
        ax.bar(ys,z,i,zdir='y',align='edge')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Train_data Nr.')
    ax.set_zlabel('Attention')

    plt.show()
def histshow(arr):
    plt.style.use('ggplot')
    plt.hist(arr, # 绘图数据
        bins = len(arr), # 指定直方图的条形数为20个
        color = 'steelblue', # 指定填充色
        edgecolor = 'k', # 指定直方图的边界色
        label = '直方图' )# 为直方图呈现标签
    plt.tick_params(top='off', right='off')
    plt.legend()
# 显示图形
    plt.show()
    



