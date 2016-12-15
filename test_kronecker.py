#### Test kronecker functions #####
# @author Melanie Ducoffe #########
# @date 01/06/2016 ################
###################################

import numpy as np
from architecture_new import Architecture
from covering.retrieve_params import (build_intermediate_var_labelled,
			     build_intermediate_var_unlabelled)

from covering.kronecker_factor import (kfac_labelled, kfac_unlabelled, blocks_Fisher_P,
				blocks_Fisher_Q, kfac_query)
from covering.covering import build_f_l, build_f_u

"""
def build_model():
	num_channels = 3
	image_size = (32,32)
	L_dim_conv_layers = [64,64, 64]
	L_filter_size = [(3,3), (3,3), (3,3)]
	L_pool_size = [ (2,2), (2,2), None]
	L_activation_conv = ['rectifier', 'rectifier', 'rectifier']
	L_dim_full_layers=[512, 20]
	L_activation_full=['rectifier', 'rectifier']
	prediction = 10
	prob_dropout = 0.5
	comitee_size = 5
	dropout_training = 0
	model = Architecture(image_size=image_size,
				num_channels=num_channels,
				L_dim_conv_layers=L_dim_conv_layers,
				L_activation_conv=L_activation_conv,
				L_filter_size=L_filter_size,
				L_pool_size=L_pool_size,
				L_dim_full_layers=L_dim_full_layers,
				L_activation_full=L_activation_full,
				prediction=prediction,
				prob_dropout=prob_dropout,
				comitee_size=comitee_size,
				dropout_training=dropout_training)
	model.initialize()
	return model, L_filter_size
"""
def build_model():
	# change params
	num_channels = 1
	image_size = (28,28)
	L_dim_conv_layers = [20,20, 5]
	L_filter_size = [(3,3), (5,5), (5,5)]
	L_pool_size = [ (2,2), (2,2), (2,2)]
	L_activation_conv = ['rectifier', 'rectifier', 'rectifier']
	L_dim_full_layers=[512]
	L_activation_full=['rectifier']
	prediction = 10
	prob_dropout = 0.5
	comitee_size = 3
	dropout_training = 0
	model = Architecture(image_size=image_size,
				num_channels=num_channels,
				L_dim_conv_layers=L_dim_conv_layers,
				L_activation_conv=L_activation_conv,
				L_filter_size=L_filter_size,
				L_pool_size=L_pool_size,
				L_dim_full_layers=L_dim_full_layers,
				L_activation_full=L_activation_full,
				prediction=prediction,
				prob_dropout=prob_dropout,
				comitee_size=comitee_size,
				dropout_training=dropout_training)
	model.initialize()
	return model, L_filter_size

def test_build_f_l():
	model, filters_info = build_model()
	f = build_f_l(model, filters_info)
	x_value = np.random.ranf((100, 3, 32, 32))
	y_value = np.ones((100,))
	dico = f(x_value, y_value)


def test_kfac_labelled():
	X = np.random.ranf((130, 3, 32, 32))
	Y = np.ones((64,)).astype(np.uintc)
	model, filters_info = build_model()
	f = build_f_l(model, filters_info)
	kfac_labelled(X, Y, f)


def test_build_f_u():
	model, filters_info = build_model()
	f = build_f_u(model, filters_info)
	x_value = np.random.ranf((100, 3, 32, 32))
	dico = f(x_value)


def test_kfac_unlabelled():
	X = np.random.ranf((12, 3, 32, 32))
	model, filters_info = build_model()
	f = build_f_u(model, filters_info)
	kfac_unlabelled(X, f)


def test_build_intermediate_var_labelled():
	# build model
	model = build_model()
	f = build_intermediate_var_labelled(model)
	x_value = np.random.ranf((100, 3, 32, 32))
	dico = f(x_value, np.zeros((100,), dtype=np.int))
	print dico.keys()


def test_build_intermediate_var_unlabelled():
	# build model
	model = build_model()
	f = build_intermediate_var_unlabelled(model)
	x_value = np.random.ranf((100, 3, 32, 32))
	dico = f(x_value)

def test_blocks_Fisher_P():
	model, filters_info = build_model()
	f_u = build_f_u(model, filters_info)
	f_l = build_f_l(model, filters_info)
	X_U = np.random.ranf((50, 1, 28, 28))
	X_L = np.random.ranf((100, 1, 28, 28))
	Y_L = np.asarray([np.random.randint(10) for i in range(100)]).reshape((100, 1))
	blocks_Fisher_P(X_L, Y_L, X_U, f_l, f_u, 50)

def test_blocks_Fisher_Q():
	model, filters_info = build_model()
	f_u = build_f_u(model, filters_info)
	X_U = np.random.ranf((1, 3, 32, 32)) # test only with 1 for memory consumption
	query = blocks_Fisher_Q(X_U, f_u)
	import pdb
	pdb.set_trace()

def test_kfac_query():
	model, filters_info = build_model()
	f_u = build_f_u(model, filters_info)
	X = np.random.ranf((10, 3, 32, 32))
	batch_size = len(X)
	kfac_query(X, f_u, batch_size)


def test_model():
	model, filters_info = build_model()
	


if __name__=="__main__":
	#test_kfac_unlabelled()
	#test_build_f_l()
	#test_kfac_query()
	test_model()
