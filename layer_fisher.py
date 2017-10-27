#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 11:17:17 2017

@author: annexe Fisher
"""
from keras.models import Model, Sequential
from keras.layers import Dense
import keras.backend as K
from keras.layers.merge import Concatenate
from keras.engine import InputSpec, Input, Layer
import numpy as np
from appendix import keras_expansion_op
import re


class KFAC(object):
    
    def __init__(self, model):
        self.model = model
        self.x = model.get_input_at(0)
        self.nb_class = model.get_output_shape_at(0)[-1]
        self.y = K.placeholder((None, self.nb_class))
        self.target = K.cast(self.y, 'int64')
        y_pred = self.model.output
        self.cost = K.categorical_crossentropy(y_pred, self.target.flatten())
        self.fisher_functions=None
        if self.cost.ndim == 2:
            # reshape into a vector
            self.cost = self.cost.flatten()
        # pretraining on the layers to get inputs and outputs
        
        # or split between trainable and not trainables 
        
    def priv_KFAC_Dense(self, layer):
        
        s_vec = layer.get_output_at(0)
        def tau(i):
            # to check
            output = K.gradients(self.cost[i], s_vec)[i]
            output = output.flatten()[:,None]
            return K.dot(output, output.T)[None, :, :]
        
        a_vec = layer.get_input_at(0)
        def phi(i):
            output = a_vec[i].flatten()[:,None]
            if layer.use_bias:
                output_bias = K.ones((1,1))
                output = K.concatenate([output, output_bias], axis=0)
            return K.dot(output, output.T)[None,:,:]
            
            
        batch_size = self.model.input.shape[0]
        index = K.arange(batch_size)
        
        #import pdb; pdb.set_trace()
        fisher_tau, _ = K.theano.scan(fn=tau,
                                      outputs_info=None,
                                      sequences=[index],
                                      non_sequences=[])
        
        fisher_phi, _ = K.theano.scan(fn=phi,
                                  outputs_info=None,
                                  sequences=[index],
                                  non_sequences=[])

        # compile
        f = K.function([K.learning_phase(), self.model.input, self.y], [fisher_phi, fisher_tau])
        
        return f
    
    def priv_Fisher_BN(self, layer):
        
        beta = layer.beta
        gamma = layer.gamma
        
        def func(i):
            grad_beta = K.gradients(self.cost[i], beta).flatten()
            grad_gamma = K.gradients(self.cost[i], gamma).flatten()
            output = K.concatenate((grad_beta, grad_gamma))[:,None]
            return K.dot(output, output.T)[None, :, :]
        
        batch_size = self.model.input.shape[0]
        index = K.arange(batch_size)
        fisher_bn, _ = K.theano.scan(fn=func,
                                     outputs_info=None,
                                     sequences=[index],
                                     non_sequences=[])
        
        f = K.function([K.learning_phase(), self.model.input, self.y], [fisher_bn])
        
        return f
    
    def priv_conv_bias(self, layer):
        
        bias = layer.bias
        
        def func(i):
            grad_bias = K.gradients(self.cost[i], bias).flatten()
            output = grad_bias[:,None]
            return K.dot(output, output.T)[None, :, :]
        
        batch_size = self.model.input.shape[0]
        index = K.arange(batch_size)
        fisher_conv_bias, _ = K.theano.scan(fn=func,
                                     outputs_info=None,
                                     sequences=[index],
                                     non_sequences=[])
        
        f = K.function([K.learning_phase(), self.model.input, self.y], [fisher_conv_bias])
        
        return f
    
    
    def priv_KFAC_Conv(self, layer):
        input_conv = layer.get_input_at(0)
        output_conv = layer.get_output_at(0)
        
        assert output_conv.ndim==4 and input_conv.ndim==4, ('wrong number of dimensions, \
        expected input and output to have dim 4 but got instead %d', output_conv.ndim)
        def tau(i):
            grad_s = K.gradients(self.cost[i], output_conv)[i]
            
            # depending on the input shape ?
            if K.image_dim_ordering() == "th":
                (_, J, w, h) = layer.get_output_shape_at(0)
            if K.image_dim_ordering() == "tf":
                (_, w, h, J) = layer.get_output_shape_at(0)
                grad_s = grad_s.transpose((2, 0, 1))

            output = grad_s.reshape((J, w*h))
            return K.dot(output, output.T)[None, :, :]
        
        mat_A = layer.get_input_at(0)
        delta = layer.kernel_size
        input_shape = layer.get_input_shape_at(0)
        expansion_mat = keras_expansion_op(mat_A, delta, input_shape)
        
        def phi(i):
            output = expansion_mat[i]
            return K.dot(output.T, output)[None, :, :]
            
            
        
        batch_size = self.model.input.shape[0]
        index = K.arange(batch_size)
        
        #import pdb; pdb.set_trace()
        fisher_tau, _ = K.theano.scan(fn=tau,
                                      outputs_info=None,
                                      sequences=[index],
                                      non_sequences=[])
        
        fisher_phi, _ = K.theano.scan(fn=phi,
                                      outputs_info=None,
                                      sequences=[index],
                                      non_sequences=[])
        
        f = K.function([K.learning_phase(), self.model.input, self.y], [fisher_phi, fisher_tau])
        
        return f
                

    def build_Fisher(self):
        
        if not(self.fisher_functions is None):
            return self.fisher_functions
        layers = self.model.layers
        fisher_functions = {}
        for layer in layers:
            if re.match('dense', layer.name):
                f = self.priv_KFAC_Dense(layer)
                fisher_functions[layer.name] = f
            if re.match('conv2d', layer.name):
                f_kernel = self.priv_KFAC_Conv(layer)
                f_bias = self.priv_conv_bias(layer)
                fisher_functions[layer.name+'_kernel']=f_kernel
                fisher_functions[layer.name+'_bias']=f_bias            
            if re.match('batch_normalization', layer.name):
                f_bn = self.priv_Fisher_BN(layer)
                fisher_functions[layer.name]=f_bn  
        self.fisher_functions = fisher_functions
        return fisher_functions
