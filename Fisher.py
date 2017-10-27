#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:43:00 2017

@author: mducoffe
"""
import re
from keras.models import Model, Sequential
import numpy as np
from annexe import build_fisher, build_queries # ???
from annexe_bis import build_fisher_biases # ???
import pickle as pkl
import keras.backend as K
import theano.tensor as T

class Fisher(object):
    
    def __init__(self, network):
        self.network = network
        self.built=False

        self.f = None
        #self.stochastic_layers = {} 
        #self.filter_layer()
        
        #layers = self.network.layers
        #for layer in layers:
        #    if layer.name in self.stochastic_layers:
        #        tmp = self.stochastic_layers[layer.name]
        #        setattr(layer, tmp[0], tmp[2])
        
        layers = self.network.layers
        intermediate_layers_input = []
        intermediate_layers_output = []
        for layer in layers:
            if re.match('merge_(.*)', layer.name):
                intermediate_layers_input.append(layer.input[0])
                intermediate_layers_output.append(layer.output)
            else:
                intermediate_layers_input.append(layer.input)
                intermediate_layers_output.append(layer.output)

        self.intermediate_input = Model(self.network.input, 
                                        [input_ for input_ in intermediate_layers_input])
        self.intermediate_output = Model(self.network.input, 
                                         [output_ for output_ in intermediate_layers_output])
        self.f = None
        
        self.dico_fisher = None
        
    def build(self):
        if self.built:
            return True
        
        f = build_fisher(self.network, self.intermediate_input, self.intermediate_output)
        self.f = f
        
    def build_Fisher_dataset(X, Y):
        #psi, tau = self.f(X, Y)
        raise NotImplementedError()
        
    def build_Fisher_samples(x, y):
        raise NotImplementedError()

    
    

        
        

