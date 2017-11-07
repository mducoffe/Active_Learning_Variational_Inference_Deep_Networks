#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 09:08:05 2017

@author: mducoffe

functions related to the usage of Fisher matrices
"""
import numpy as np
import keras.backend as K


def fisher_set(fisher_functions, data, y_data):
    dico_fisher={}
    # get labels
    N_data = data.shape[0]
    batch_size = 16
    batch_index = np.round(np.linspace(0,N_data,N_data/batch_size+1)).astype('uint64')
    for key in fisher_functions.keys():
        # split the data into batches
        fisher_list = []
        mean_fisher = []
        nb_elem = 0
        for index_0, index_1 in zip(batch_index[:-1], batch_index[1:]):
            batch_data=data[index_0:index_1]
            batch_y = y_data[index_0:index_1]
            #print((key, index_0))
            tmp_fisher_list = fisher_functions[key]([0, batch_data, batch_y])
            
            ####
            if len(mean_fisher)==0:
                # first elem
                if len(tmp_fisher_list)==1:
                    mean_fisher = [ np.sum(tmp_fisher_list[0], axis=0)]
                else:
                    mean_fisher = [np.sum(tmp_fisher_list[0], axis=0), np.sum(tmp_fisher_list[1], axis=0)]
                nb_elem += tmp_fisher_list[0].shape[0]
            """
            if len(fisher_list)==0:
                if len(tmp_fisher_list)==1:
                    fisher_list = tmp_fisher_list
                else:
                    fisher_list=[ [tmp_fisher_list[0]], [tmp_fisher_list[1]] ]
                
            else:
                if len(tmp_fisher_list)==1:
                    fisher_list.append(tmp_fisher_list[0])
                else:
                    fisher_list[0].append(tmp_fisher_list[0])
                    fisher_list[1].append(tmp_fisher_list[1])
            """
            
        """
        if len(tmp_fisher_list)==1:
            fisher_list = [np.concatenate(fisher_list)]
            mean_fisher = [np.sum(fisher_list[0], axis=0)]
            nb_elem = len(fisher_list[0])
        else:
            fisher_list = [np.concatenate(elem) for elem in fisher_list]
            mean_fisher = [np.sum(fisher_list[0], axis=0), np.sum(fisher_list[1], axis=0)]
            nb_elem = len(fisher_list[0])
        dico_fisher[key]=[]
        import pdb; pdb.set_trace()
        for elem_fisher in mean_fisher:
            #dico_fisher[key].append(np.mean(elem_fisher, axis=0))
            dico_fisher[key].append(mean_fisher/nb_elem)
        """

        dico_fisher[key]=[]
        for elem_fisher in mean_fisher:
            dico_fisher[key].append(1.*elem_fisher/nb_elem)
            
    return dico_fisher

def invert_fisher(dico_fisher):
    
    dico_inv_fisher={}
    
    def inv_matrix_function(mat):
        if mat.ndim ==3:
            mat = mat[0]
        n = mat.shape[0]
        if mat.shape[0]!=mat.shape[1]:
            print('problem')
            import pdb; pdb.set_trace()
        assert mat.shape[0]==mat.shape[1], 'cannot invert a non square matrix'
        noise = np.diag([10**(-20)]*n)
        return np.linalg.inv(mat+noise)[None,:,:]
    
    for key in dico_fisher.keys():
        fisher_list = dico_fisher[key]
        dico_inv_fisher[key]=[]
        if len(fisher_list)==1:
            elem_fisher = fisher_list[0]
            inv_matrix = inv_matrix_function(elem_fisher)
            dico_inv_fisher[key].append(inv_matrix)
            
        else:
            elem_fisher_phi, elem_fisher_tau = fisher_list
            inv_matrix_phi = inv_matrix_function(elem_fisher_phi)
            inv_matrix_tau = inv_matrix_function(elem_fisher_tau)
            dico_inv_fisher[key].append(inv_matrix_phi)
            dico_inv_fisher[key].append(inv_matrix_tau)
            
            
    return dico_inv_fisher

def multiply():
    A = K.theano.tensor.tensor4()
    B = K.theano.tensor.tensor3()
    
    def func(i,A):
        C = K.dot(A[i,0], B[0])
        return K.theano.tensor.nlinalg.trace(C)
    
    N = A.shape[0]
    result, _ = K.theano.scan(fn=func, outputs_info=None,
                              sequences=[K.arange(N)],
                              non_sequences=[A])
    
    """
    def function(A, B):
        C = K.dot(A, B)
        return K.theano.tensor.nlinalg.trace(C)
    
    result,_ = K.theano.scan(fn=function)
    """
    func = K.function([A, B], result) # TO_DO
    return func

def active_selection_fisher(model, fisher_functions, multiply, labelled_data, unlabelled_data, nb_data):
    
    #select a subset of data
    n = min(1000, len(unlabelled_data[0]))
    subset_index = np.random.permutation(len(unlabelled_data[0]))
    subset_data = unlabelled_data[0][subset_index[:n]]
    subset_label = unlabelled_data[0][subset_index[:n]] # UNUSED
    
    #index_active_query = np.random.permutation(len(subset_index))[:nb_data]
    #index_active_unlabelled = np.random.permutation(len(subset_index))[nb_data:]
    index_active_query, index_active_unlabelled = active_selection_fisher_priv(model, fisher_functions, multiply, labelled_data, (subset_data, subset_label), nb_data)
    index_query = subset_index[index_active_query]
    index_unlabelled = np.concatenate( (subset_index[index_active_unlabelled], subset_index[n:]))
    
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])


def active_selection_fisher_priv(model, fisher_functions, multiply, labelled_data, unlabelled_data, nb_data):
    
    print('deb active_selection_fisher_priv')
    fisher_labelled_dico = fisher_set(fisher_functions, labelled_data[0], labelled_data[1])
    inv_fisher_labelled = invert_fisher(fisher_labelled_dico)
    layers_name = fisher_functions.keys()
    # get unlabelled labels
    y_unlabelled = np.argmax(model.predict(unlabelled_data[0]), axis=1).flatten() #shape = (minibatch,)
    N_unlabelled = len(y_unlabelled)
    batch_size = 32
    batch_index = np.round(np.linspace(0,N_unlabelled,batch_size)).astype('uint64')
    nb_output = model.get_output_shape_at(0)[-1]
    y_data = np.zeros((unlabelled_data[0].shape[0], nb_output))
    for i in range(unlabelled_data[0].shape[0]):
        y_data[i,y_unlabelled[i]]=1
    
    fisher_unlabelled_data = {}
    for key in layers_name:
        list_key = []
        for index_0, index_1 in zip(batch_index[:-1], batch_index[1:]):
            batch_data = unlabelled_data[0][index_0:index_1]
            batch_y = y_data[index_0:index_1]
            tmp_index = fisher_functions[key]([1,batch_data, batch_y])
            
            if len(list_key)==0:
                if len(tmp_index)==1:
                    list_key = tmp_index
                else:
                    list_key=[[tmp_index[0]], [tmp_index[1]]]
            else:
                if len(tmp_index)==1:
                    list_key.append(tmp_index[0])
                else:
                    list_key[0].append(tmp_index[0])
                    list_key[1].append(tmp_index[1])
        #tmp = fisher_functions[key]([1,unlabelled_data[0], y_data])
        if len(tmp_index)==1:
            tmp = [ np.concatenate(list_key)]
        else:
            tmp = [ np.concatenate(elem) for elem in list_key]
        fisher_unlabelled_data[key]=tmp
    for key in layers_name:
        
        elem_fisher_unlabelled = fisher_unlabelled_data[key]
        if len(elem_fisher_unlabelled)==1:
            N_elem = elem_fisher_unlabelled[0].shape[0]
            fisher_unlabelled_data[key] = [ np.ones((N_elem, 1)),
                                  multiply([elem_fisher_unlabelled[0], inv_fisher_labelled[key][0]])]
            
        else:
            phi_U, tau_U = fisher_unlabelled_data[key]
            phi_1_L, tau_1_L = inv_fisher_labelled[key]
            # need to split for gpu memory
            fisher_data_tmp = [ [], []]
            for index_0, index_1 in zip(batch_index[:-1], batch_index[1:]):
                batch_phi = phi_U[index_0:index_1]
                batch_tau = tau_U[index_0:index_1]
                tmp_phi = multiply([batch_phi, phi_1_L])
                tmp_tau = multiply([batch_tau, tau_1_L])
                fisher_data_tmp[0].append(tmp_phi)
                fisher_data_tmp[1].append(tmp_tau)
                
            fisher_unlabelled_data[key]= [ np.concatenate(fisher_data_tmp[0]), np.concatenate(fisher_data_tmp[1])]
            #fisher_unlabelled_data[key] = [ multiply([phi_U, phi_1_L]), multiply([tau_U, tau_1_L])]
            
             
    coeff_phi = dict([(key, [elem for elem in fisher_unlabelled_data[key][0]]) for key in fisher_unlabelled_data.keys()])
    coeff_tau = dict([(key, [elem for elem in fisher_unlabelled_data[key][1]]) for key in fisher_unlabelled_data.keys()])
    alphas = dict([(key, len(labelled_data[0])*np.prod([elem.shape[-1] for elem in inv_fisher_labelled[key]])) for key in coeff_phi.keys()])
    index_query = greedy_selection(coeff_phi, coeff_tau, alphas, nb_data)
    # sort indices
    index_query = np.sort(index_query)[::-1]
    index_unlabelled = range(len(unlabelled_data[0]))

    for i in index_query:
        if i >=len(index_unlabelled):
            import pdb; pdb.set_trace()
        index_unlabelled.pop(i)
        
    print('end active_selection_fisher_priv')
    return index_query, index_unlabelled
    """
    return (unlabelled_data[0][index_query], unlabelled_data[1][index_query]), \
           (unlabelled_data[0][index_unlabelled], unlabelled_data[1][index_unlabelled])
    """
    
def greedy_selection(coeff_phi, coeff_tau, alphas, batch_size):
    
    index=[]
    keys = coeff_phi.keys()
    phi = dict([(key,0) for key in keys])
    tau = dict([(key,0) for key in keys])
    N_data = len(coeff_phi[keys[0]])
    while (len(index)<batch_size or len(index)>=N_data):
        new_scores={}
        for key in keys:
            alpha = alphas[key]
            new_scores_key = [((phi[key] + phi_i)*(tau[key]+tau_i) + alpha*(phi_i+tau_i)) \
                    for (phi_i, tau_i) in zip(coeff_phi[key], coeff_tau[key])]
        
            new_scores[key]= new_scores_key
        scores = np.sum(new_scores.values(), axis=0)
        # re
        # select data which minimize the score
        i = np.argmin(scores)
        if i in index:
            print('error, selecting twice a sample')
            import pdb; pdb.set_trace()
        for key in keys:
            phi[key]+=coeff_phi[key][i]
            tau[key]+=coeff_tau[key][i]
            coeff_phi[key][i]= np.inf
            coeff_tau[key][i] = np.inf

        index.append(i)
        
    return index
    
    
# 'conv2d_3_kernel', 'dense_5', 'conv2d_4_bias', 'conv2d_3_bias', 'conv2d_4_kernel', 'dense_6']
        
    
    
    

# TO DO
# multiply function on GPU
# trace
# coefficients
# selection
