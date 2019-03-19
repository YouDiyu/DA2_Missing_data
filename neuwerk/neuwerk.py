# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 17:48:29 2018

@author: You
"""
from keras import layers
from keras import Input
from keras.models import Model
Minput = Input(shape=(None,), dtype='int32', name='posts')
