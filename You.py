# -*- coding: utf-8 -*-
"""
Created on Mon Nov 26 16:11:58 2018

@author: You
"""

from keras.engine.topology import Layer

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.utils.generic_utils import to_list
from keras.utils.generic_utils import unpack_singleton
from keras import layers
from keras.layers import GRU
from keras.layers import Dense

class data_preprocess():
    
    '''
    data zero_mean_nomalization and replace the missing data
    '''
    #if x is float return x else retun None
    def floatornone(self,x):
        try:
            x=float(x)
        except ValueError:
            x=None
        return x
    
    def generate_mask(self,train_data):
        mask=np.minimum(train_data,0)
        mask=np.maximum(mask,1)
        mask[np.isnan(mask)]=0
        return mask
    
    def mask_weighting(self,mask):
        maskweights=np.sum(mask,axis=-1)
        maskweights_matrix=(mask.T/maskweights*mask.shape[-1]).T
        return maskweights_matrix
    
    def none_to_zero(self,train_data):
        train_data_zero=train_data.copy()
        train_data_zero[np.isnan(train_data)]=0
        return train_data_zero
    def zero_mean_normalization(self,train_data,weighting=False):
        #zero-mean-normalization
        mask_binary=self.generate_mask(train_data)
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
        processed_data=(train_data_mean-mean)/(std+K.epsilon())
        if weighting:
            processed_data=processed_data*self.mask_weighting(mask_binary)
            return processed_data
        else:
            return processed_data
def zero_mean_normalization(self,train_data,weighting=False):
        #zero-mean-normalization
        mask_binary=self.generate_mask(train_data)
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
        processed_data=(train_data_mean-mean)/(std+K.epsilon())+0.5
        if weighting:
            processed_data=processed_data*self.mask_weighting(mask_binary)
            return processed_data
        else:
            return processed_data
class GRU_M(GRU):
    def __init__(self,units, **kwargs):
        super(GRU_M,self).__init__(units,**kwargs)
        self.supports_masking = True
    def compute_mask(self, inputs, mask=None):
        inputs=tf.abs(inputs)
        mask_bool=inputs>0
        mask=tf.cast(mask_bool,dtype=K.floatx())
        return mask
    

    
class AttentionLayer_2D(Layer):
    """
    input_shape:batch_size,time_steps,seq_len
    output_shape = input_shape
    
    """
    def __init__(self,supmask=False,**kwargs):
        self.supmask=supmask
        self.att=[]
        self.z=[]
        super(AttentionLayer_2D,self).__init__(**kwargs)
    def build(self,input_shape):
        #includ time steps
        assert len(input_shape)==2
        self.W=self.add_weight(name='att_weight',
                               shape=(input_shape[1],input_shape[1]),
                               initializer='uniform',
                               trainable=True)
        self.b=self.add_weight(name='att_bias',
                               shape=(input_shape[1],),
                               initializer='uniform',
                               trainable=True)
        super(AttentionLayer_2D,self).build(input_shape)


    def call(self,inputs):
        #batch_size,Seq_len,time_steps (None,32,5)
        
        mask_bool=tf.abs(inputs)>0
        inputs=tf.where(mask_bool,tf.cast(inputs,tf.float32),tf.cast(mask_bool,dtype=K.floatx()))
        x=inputs
        if self.supmask:
            previous_mask=_collect_previous_mask(inputs)
            x=x*previous_mask
        #batch_size,Seq_len,time_steps
        self.z=K.dot(x,self.W)+self.b
        activation=K.tanh(self.z)        
        expatt=K.exp(activation)

        self.att=expatt/K.cast(K.sum(expatt,axis=1,keepdims=True)+K.epsilon(),K.floatx())       
        #att=K.softmax(activation)
        #(None,5,32)
        outputs=self.att*x
        #outputs = K.sum(outputs, axis=-1)
        #outputs batch_size,time_steps,Seq_len
        return outputs
    def compute_output_shape(self,input_shape):
        return input_shape
        #return input_shape
    def get_attentionwerte(self):
        paras=self.att
        return K.batch_get_value(paras)
    def compute_mask(self, inputs, mask=None):
        return None


class AttentionLayer_3D(Layer):
    """
    input_shape:batch_size,time_steps,seq_len
    output_shape = input_shape
    
    """
    def __init__(self,supmask=False,**kwargs):
        self.supmask=supmask
        self.att=[]
        super(AttentionLayer_3D,self).__init__(**kwargs)
    def build(self,input_shape):
        #includ time steps
        #assert len(input_shape)==3 （8，5，5）
        self.W=self.add_weight(name='att_weight',
                               shape=(input_shape[2],input_shape[1],input_shape[1]),
                               initializer='uniform',
                               trainable=True)
                                    #(8,5)
        self.b=self.add_weight(name='att_bias',
                               shape=(input_shape[2],input_shape[1],),
                               initializer='uniform',
                               trainable=True)

        super(AttentionLayer_3D,self).build(input_shape)


    def call(self,inputs):
        #32,5,8
        #batch_size,Seq_len,time_steps (None,32,5)#(32,8,5)
        
        x=K.permute_dimensions(inputs,(0,2,1))
        #bs 8 1 5
        previous_mask=_collect_previous_mask(inputs)
        mask=K.permute_dimensions(previous_mask,(0,2,1))
        x=x*mask
        def scan_dot(vol,var):
            return K.sum(tf.matmul(K.expand_dims(var,axis=1),self.W),axis=1)+self.b
        
        z=tf.scan(fn=scan_dot,elems=x)#bs,8,5
        activation=K.tanh(z)        
        expatt=K.exp(activation)
        self.att=expatt/K.cast(K.sum(expatt,axis=2,keepdims=True)+K.epsilon(),K.floatx())       
        #att=K.softmax(activation)
        #(None,5,32)
        outputs=self.att*x
        #/tf.cast(tf.count_nonzero(mask),tf.float32)*tf.cast(K.int_shape(mask)[-1]*K.int_shape(mask)[-2],tf.float32)
        outputs = K.sum(outputs, axis=1)
        #outputs batch_size,time_steps,Seq_len
        return outputs

    def compute_output_shape(self,input_shape):
        return input_shape[0], input_shape[1]
        #return input_shape
    def get_attentionwerte(self):
        paras=self.att
        return K.batch_get_value(paras)
    def compute_mask(self, inputs, mask=None):
        return None





class AttentionLayer(Layer):
    """
    input_shape:batch_size,time_steps,seq_len
    output_shape = input_shape
    
    """
    def __init__(self,supmask=False,**kwargs):
        self.supmask=supmask
        self.att=[]
        super(AttentionLayer,self).__init__(**kwargs)
    def build(self,input_shape):
        #includ time steps
        #assert len(input_shape)==3
        self.W=self.add_weight(name='att_weight',
                               shape=(input_shape[1],input_shape[1]),
                               initializer='uniform',
                               trainable=True)

        self.b=self.add_weight(name='att_bias',
                               shape=(input_shape[1],),
                               initializer='uniform',
                               trainable=True)

        super(AttentionLayer,self).build(input_shape)


    def call(self,inputs):
        #batch_size,Seq_len,time_steps (None,32,5)32,5,8>32,8,5
        x=K.permute_dimensions(inputs,(0,2,1))
        ###
        previous_mask=_collect_previous_mask(inputs)
        mask=K.permute_dimensions(previous_mask,(0,2,1))
        x=x*mask
        ###
        #batch_size,Seq_len,time_steps
        z=K.dot(x,self.W)+self.b
        activation=K.tanh(z)        
        expatt=K.exp(activation)
        
        if self.supmask:
            previous_mask=_collect_previous_mask(inputs)
            mask=K.permute_dimensions(previous_mask,(0,2,1))
            expatt=expatt*mask
        self.att=expatt/K.cast(K.sum(expatt,axis=2,keepdims=True)+K.epsilon(),K.floatx())       
        #att=K.softmax(activation)
        #(None,5,32)
        outputs=K.permute_dimensions(self.att*x,(0,2,1))
        outputs = K.sum(outputs, axis=2)
        #outputs batch_size,time_steps,Seq_len
        return outputs

    def compute_output_shape(self,input_shape):
        return input_shape[0], input_shape[1]
        #return input_shape
    def get_attentionwerte(self):
        paras=self.att
        return K.batch_get_value(paras)
    def compute_mask(self, inputs, mask=None):
        return None
def _collect_previous_mask(input_tensors):
    """Retrieves the output mask(s) of the previous node.

    # Arguments
        input_tensors: A tensor or list of tensors.

    # Returns
        A mask tensor or list of mask tensors.
    """
    input_tensors = to_list(input_tensors)
    masks = []
    for x in input_tensors:
        if hasattr(x, '_keras_history'):
            inbound_layer, node_index, tensor_index = x._keras_history
            node = inbound_layer._inbound_nodes[node_index]
            mask = node.output_masks[tensor_index]
            masks.append(mask)
        else:
            masks.append(None)
    return unpack_singleton(masks)    

        


class activeDrop(Layer):
    '''
    Applies ActiveDropout to the input.
    active drop the Missing data.
    #Arguments
        rate=None:only drop the missing data 
        Remaining Data weighting with 1/(1-Loss Rate)
        0.<rate<1.:randomly setting a fraction `rate` of 
        Remaining Data to 0 and weighting with 1/(1-Rate)
        
    '''
    def __init__(self, rate=None,  **kwargs):
        super(activeDrop,self).__init__(**kwargs)
        self.rate = rate

    def call(self,inputs,training=None):
        #None to zero
        #tf.where(inputs,a,b) inputs bool, true-a,false-b
        #inputs=tf.cast(inputs,tf.float32)
        inputs=K.dropout(inputs,0.0001,None,None)
        mask_bool=tf.abs(inputs)>0
        inputs=tf.where(mask_bool,tf.cast(inputs,tf.float32),tf.cast(mask_bool,dtype=K.floatx()))

        count_zeros=tf.count_nonzero(inputs,-1)
        inputs_T=tf.transpose(inputs)
        inputs_gewichted_T=tf.div(tf.cast(inputs_T,tf.float32),
                                  (tf.cast(count_zeros,tf.float32))+K.epsilon())
        inputs_gewichted_T*=K.int_shape(inputs)[-1]
        inputs_gewichted=tf.transpose(inputs_gewichted_T)

        if self.rate==None:
            return K.in_train_phase(inputs_gewichted, inputs_gewichted, training=training)
        if 0. < self.rate < 1.:
            def dropped_inputs():
                return K.dropout(inputs_gewichted, self.rate, None,None)
            return K.in_train_phase(dropped_inputs, inputs,
                                    training=training)
        return inputs
        

    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        config = {'rate': self.rate,}
        base_config = super(activeDrop, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





