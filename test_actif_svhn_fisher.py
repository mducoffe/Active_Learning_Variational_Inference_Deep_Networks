#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4 20:06:51 2017

@author: mducoffe

fisher batch selection on cifar10
"""

import sys
sys.path.append('/home/ducoffe/Documents/Code/keras_fisher')

import numpy as np
import sklearn.metrics as metrics
import argparse
import keras
import keras.utils.np_utils as kutils
from keras.datasets import cifar10
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from snapshot import SnapshotCallbackBuilder
import csv
from contextlib import closing
import os
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Activation
from keras.layers import Conv2D, MaxPooling2D

from layer_fisher import KFAC
from fisher_func import active_selection_fisher, multiply
import pickle
import gc

#%%
import resource
from keras.callbacks import Callback
class MemoryCallback(Callback):
    def on_epoch_end(self, epoch, log={}):
        print(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


#%%

def build_data(num_sample=100, img_rows=32, img_cols=32):

    import scipy.io as io
    
    dico_train = io.loadmat(os.path.join('./svhn','train_32x32.mat'))
    x_train = dico_train['X']
    x_train = x_train.transpose((3, 0,1, 2))
    y_train = dico_train['y'] -1
    
    dico_test = io.loadmat(os.path.join('./svhn','test_32x32.mat'))
    x_test = dico_test['X']
    x_test = x_test.transpose((3,0,1, 2))
    y_test = dico_test['y'] -1

    
    x_train = x_train.astype('float32')
    x_train /= 255.0
    x_test = x_test.astype('float32')
    x_test /= 255.0
    
    if K.image_dim_ordering() == "th":
        x_train = x_train.transpose((0,3,1,2))
        x_test = x_test.transpose((0,3,1,2))
    
    y_train = kutils.to_categorical(y_train)
    y_test = kutils.to_categorical(y_test)
    N = len(x_train)
    index = np.random.permutation(N)
    x_train = x_train[index]
    y_train = y_train[index]
    
    x_L = x_train[:num_sample]; y_L = y_train[:num_sample]
    x_U = x_train[num_sample:]; y_U = y_train[num_sample:]
    return (x_L, y_L), (x_U, y_U), (x_test, y_test)

#%%
def build_model(num_classes=10):

    img_rows, img_cols = 32, 32
    
    if K.image_dim_ordering() == "th":
        input_shape = (3, img_rows, img_cols)
    else:
        input_shape = (img_rows, img_cols, 3)
        
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["acc"])
    return model

#%%
def active_training(labelled_data,
                    batch_size=64, epochs=100, repeat=3):
    
    x_L, y_L = labelled_data 
    
    # split into train and validation
    index = np.random.permutation(len(y_L))
    N = len(index)
    x_L = x_L[index]; y_L = y_L[index]
    
    
    
    # train and valid generator
    generator_train = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=5./32,
                                   height_shift_range=5./32,
                                   horizontal_flip=True)

    generator_train.fit(x_L, seed=0, augment=True)
    batch_train = min(batch_size, len(x_L))
    steps_per_epoch_train = int(N/batch_train)
    tmp = generator_train.flow(x_L, y_L, batch_size=batch_size)
    values = [tmp.next() for i in range(steps_per_epoch_train)]
    x_L = np.concatenate([value[0] for value in values])
    y_L = np.concatenate([value[1] for value in values])
    
    earlyStopping=keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')
    
    best_model = None
    best_loss = np.inf
    for i in range(repeat):
        model = build_model()
        
        hist = model.fit(x_L, y_L, 
             batch_size=batch_train, epochs=epochs,
             callbacks=[earlyStopping],
             shuffle=True,
             validation_split=0.2,
             verbose=0)
        
        
        loss, acc = model.evaluate(x_L, y_L, verbose=0)
        if loss < best_loss:
            best_loss = loss;
            best_model = model
        del model
        del hist
        del loss
        del acc
        i=gc.collect()
        while(i!=0):
            i=gc.collect()
        

    return best_model
#%%

def copy_weights_from(model_from, model_to):
    # model1 and model2 have the exact same architecture
    layers_from = model_from.layers
    layers_to = model_to.layers
    for layer_from, layer_to in zip(layers_from, layers_to):
        if layer_from.trainable_weights:
            weights_from = layer_from.trainable_weights
            weights_to = layer_to.trainable_weights
            for w_from, w_to in zip(weights_from, weights_to):
                w_to.set_value(w_from.get_value())
            del(weights_from)
    return model_to

            

#%%
def evaluate(model, percentage, test_data, nb_exp, repo, filename):
    x_test, y_test = test_data
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    
    with closing(open(os.path.join(repo, filename), 'a')) as csvfile:
        # TO DO
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([str(nb_exp), str(percentage), str(loss), str(acc)])
        
def fisher_selection(model, fisher_functions, multiply,
                     labelled_data, unlabelled_data, nb_data):

    index_permut = np.random.permutation(len(unlabelled_data[0]))
    """
    query = (unlabelled_data[0][index_permut[:nb_data]], unlabelled_data[1][index_permut[:nb_data]])
    unlabelled_pool = (unlabelled_data[0][index_permut[nb_data:]], unlabelled_data[1][index_permut[nb_data:]])
    """
    query, unlabelled_data = active_selection_fisher(model,
                                                     fisher_functions, 
                                                     multiply, 
                                                     labelled_data, 
                                                     unlabelled_data, 
                                                     nb_data)
    return query, unlabelled_data
    
    
    #return query, unlabelled_pool

#%%
def get_weights(model):
    layers = model.layers
    weights=[]
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            weights+=[elem.get_value() for elem in weights_layer]
    return weights
    
def load_weights(model, weights):
    layers = model.layers
    index=0
    for layer in layers:
        if layer.trainable_weights:
            weights_layer = layer.trainable_weights
            for elem in weights_layer:
                elem.set_value(weights[index])
                index+=1
    return model
                
                
def loading(repo, filename, num_sample):
    # check if file exists
    model=build_model()
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    if (os.path.isfile(os.path.join(repo, f_weights)) and \
        os.path.isfile(os.path.join(repo, f_l_data)) and \
        os.path.isfile(os.path.join(repo, f_u_data)) and \
        os.path.isfile(os.path.join(repo, f_t_data))):
        
        
        
        with closing(open(os.path.join(repo, f_weights), 'rb')) as f:
            weights = pickle.load(f)
            model = load_weights(model, weights)
            
        with closing(open(os.path.join(repo, f_l_data), 'rb')) as f:
            labelled_data = pickle.load(f)   
            
        with closing(open(os.path.join(repo, f_u_data), 'rb')) as f:
            unlabelled_data = pickle.load(f) 
            
        with closing(open(os.path.join(repo, f_t_data), 'rb')) as f:
            test_data = pickle.load(f)
    else:
        labelled_data, unlabelled_data, test_data = build_data(num_sample=num_sample)
    
    return model, labelled_data, unlabelled_data, test_data
    
def saving(model, labelled_data, unlabelled_data, test_data, repo, filename):
    weights = get_weights(model)
    #data = (weights, labelled_data, unlabelled_data, test_data)
    
    filename = filename.split('.pkl')
    f_weights = filename[0]+'_weights.pkl'
    f_l_data = filename[0]+'_labelled.pkl'
    f_u_data = filename[0]+'_unlabelled.pkl'
    f_t_data = filename[0]+'_test.pkl'
    
    with closing(open(os.path.join(repo, f_weights), 'wb')) as f:
        pickle.dump(weights, f)
    with closing(open(os.path.join(repo, f_l_data), 'wb')) as f:
        pickle.dump(labelled_data, f)
    with closing(open(os.path.join(repo, f_u_data), 'wb')) as f:
        pickle.dump(unlabelled_data, f)
    with closing(open(os.path.join(repo, f_t_data), 'wb')) as f:
        pickle.dump(test_data, f)


#%%
def active_learning_fisher(num_sample=32, percentage=0.3, 
                    nb_exp=0, repo='test', filename='test.csv'):
    
    # create a model and do a reinit function
    model, labelled_data, unlabelled_data, test_data = loading(repo, 'tmp_cifar.pkl', num_sample)
    kfac = KFAC(model)
    dico = kfac.build_Fisher()
    f_multiply = multiply()
    batch_size = 32
    nb_query=200
    percentage_data = len(labelled_data[0])
    N_pool = len(labelled_data[0]) + len(unlabelled_data[0])
    print('START')
    # load data
    i=0
    while( percentage_data<=N_pool):

        i+=1
        model_from = active_training(labelled_data, batch_size=batch_size)
        model=copy_weights_from(model_from, model)
        
        
        query, unlabelled_data = fisher_selection(model=model,
                                                  fisher_functions=dico,
                                                  multiply=f_multiply,
                                                  labelled_data=labelled_data,
                                                  unlabelled_data=unlabelled_data,  
                                                  nb_data=nb_query)
        print('SUCCEED')
        evaluate(model, percentage_data, test_data, nb_exp, repo, filename)
        # SAVE
        saving(model, labelled_data, unlabelled_data, test_data, repo, 'tmp_cifar.pkl')
        #print('SUCEED')
        #print('step B')
        i=gc.collect()
        while(i!=0):
            i = gc.collect()

        # add query to the labelled set
        labelled_data_0 = np.concatenate((labelled_data[0], query[0]), axis=0)
        labelled_data_1 = np.concatenate((labelled_data[1], query[1]), axis=0)
        labelled_data = (labelled_data_0, labelled_data_1)
        #update percentage_data
        percentage_data = len(labelled_data[0])
        
#%%
if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Active Learning on MNIST')

    parser.add_argument('--id_experiment', type=int, default=2, help='id number of experiment')
    parser.add_argument('--repo', type=str, default='.', help='repository for log')
    parser.add_argument('--filename', type=str, default='cifar_iclr_fisher', help='csv filename')
    args = parser.parse_args()
                                                                                                                                                                                                                             

    nb_exp = args.id_experiment
    repo=args.repo
    filename=args.filename
    if filename.split('.')[-1]!='csv':
        filename+='.csv'
    
    active_learning_fisher(nb_exp=nb_exp,
                    repo=repo,
                    filename=filename)
