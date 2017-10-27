#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 16:49:53 2017

@author: mducoffe

appendix : expansion operator for convolutional KFAC
"""

from keras.models import Model, Sequential
from keras.layers import Dense
import keras.backend as K
from keras.layers.merge import Concatenate
from keras.engine import InputSpec, Input, Layer
import numpy as np

def keras_expansion_op(A, delta, input_shape):
    
    if K.image_dim_ordering() == "th":
        (_, J, X, Y) = input_shape
    else:
        (_, X, Y, J) = input_shape
        A = A.transpose((0, 3, 2, 1))
    
    d_x = delta[0]/2; d_y = delta[1]/2
    
    var_x = []
    for n_x in range(d_x, X-d_x):
        var_y = []
        for n_y in range(d_y, Y-d_y):
            tmp = A[:,:, n_x -d_x:n_x+d_x+1, n_y-d_y:n_y+d_y+1]
            tmp = tmp[:,:, ::-1, ::-1, None]
            var_y.append(tmp)
        var_y = K.concatenate(var_y, axis=4)
        var_y = var_y[:,:,:,:,:,None]
        var_x.append(var_y)
    var_x = K.concatenate(var_x, axis=5)
    
    E_A = var_x.transpose((0, 5, 4, 1, 2, 3))
    batch_size = E_A.shape[0]
    coeff = 1./((X-2*d_x)*(Y-2*d_y)) # 1/sqrt(tau)
    E_A = E_A.reshape((batch_size, (X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1)))
    return coeff*E_A