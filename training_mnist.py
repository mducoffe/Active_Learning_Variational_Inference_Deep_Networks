###### TRAINING SETTING ########
###### CIFAR-10 ################
# @author Melanie Ducoffe ######
# @date 10/02/2016 #############
################################

import theano.tensor as T
import theano
import numpy as np
from blocks.utils import shared_floatx
from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from architecture_new import Architecture
from blocks.roles import WEIGHT, BIAS
from blocks.bricks import Softmax
from dropout import analyze_param_name

def get_Params(output):
	cg = ComputationGraph(output.sum())
	W = VariableFilter(roles=[WEIGHT])(cg.variables)
	B = VariableFilter(roles=[BIAS])(cg.variables)
	return W, B


def RMSProp(cost, params, lr=0.002, decay_rate=0.9):
	updates = []; updates_init=[]
	grad_params = T.grad(cost, params)
	for param, grad_param in zip(params, grad_params):
		p = shared_floatx(param.get_value() * 0.,
							   "cache_"+param.name)
		# update rule
		update_cache = decay_rate*p\
					+ (1 - decay_rate)*grad_param**2
		update_param = param  - lr*grad_param/T.sqrt(update_cache + 1e-8)
		updates.append((p, update_cache))
		updates.append((param, update_param))
		updates_init.append((p, 0*p))

	return updates, updates_init
		
def Sgd(cost, params, lr):
	grads = T.grad(cost, params)
	updates=[]
	for p,g in zip(params, grads):
		updates.append((p,p -lr*g))
	return updates, []

def Adam(cost, params, lr=0.002, b1=0.2, b2=0.001, e=1e-8):
        decay_factor = 1-e
	updates=[]
	grads=T.grad(cost, params)
	i = shared_floatx(0.,"adam_t")
	i_t = i+1
	updates.append((i,i_t))
        lr = (lr *T.sqrt((1. - (1. - b2)**i_t)) /
                         (1. - (1. - b1)**i_t))
        b1_t = 1 - (1 - b1) * decay_factor ** (i_t - 1)

	updates_init=[]
        for p,g in zip(params, grads):
            m = shared_floatx(p.get_value() * 0.,
                                                "adam_m_"+p.name)
            v = shared_floatx(p.get_value() *0.,
                                                "adam_v_"+p.name)

            m_t = b1_t*g + (1-b1_t)*g
            v_t = b2*T.sqr(g) + (1-b2)*v
            g_t = m_t/(T.sqrt(v_t)+e)
            updates.append((m,m_t))
            updates.append((v,v_t))
            updates.append((p, p-lr*g_t))
	    updates_init.append((m, 0*m))
	    updates_init.append((v, 0*v))
        return updates, updates_init

def Momentum(cost, params, lr, mu):
	updates=[]; updates_init=[]
	grad_params = T.grad(cost, params)
	for param, grad_param in zip(params, grad_params):
		velocity = shared_floatx(param.get_value()*0., "velocity_"+param.name)
		update_param = mu*velocity - lr*grad_param
		updates.append((velocity, update_param))
		updates.append((param, param+update_param))
		updates_init.append((velocity, 0*velocity))

	return updates, updates_init

def build_model():
	from blocks.config import config
	config.default_seed = np.random.randint(20)+1
	# change params
	num_channels = 1
	image_size = (28,28)
	L_dim_conv_layers = [20, 20]
	L_filter_size = [(3,3), (3,3)]
	L_pool_size = [ (2,2), (2,2)]
	L_activation_conv = ['rectifier', 'rectifier']
	L_dim_full_layers=[200, 200, 50]
	L_activation_full=['rectifier', 'rectifier', 'rectifier']
	"""
	L_dim_conv_layers = [64,4]
	L_filter_size = [(3,3), (5,5)]
	L_pool_size = [ (2,2), (2,2)]
	L_activation_conv = ['rectifier', 'rectifier']
	L_dim_full_layers=[512]
	L_activation_full=['rectifier']
	"""
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
	model.initial_state()
	return model

def training_committee_member(instance, learning_rate, train, batch_size, valid, valid_full=1.):
	#valid_full+=0.1
	x = T.tensor4('x')
	y = T.imatrix('y')
	y_prev = instance.apply(x)
	#cost = CategoricalCrossEntropy().apply(y.flatten(), y_prev).mean()
	cost = Softmax().categorical_cross_entropy(y.flatten(), y_prev).mean()
    	error = MisclassificationRate().apply(y.flatten(), Softmax().apply(y_prev)).mean()
	# take only the last parameters to avoid having the same members among the committee
	W, B = get_Params(y_prev) # take all the parameters for now !!!!
	w = W[0];
	layer_number=None
	for w_tmp in W:
		if layer_number is None:
			(_, layer_number, _) = analyze_param_name(w.name)
		(_, layer_number_tmp, _) = analyze_param_name(w_tmp.name)
		if layer_number_tmp > layer_number:
			w = w_tmp
			layer_number=layer_number_tmp
	for b_tmp in B:
		(_, layer_number_tmp, _) = analyze_param_name(b_tmp.name)
		if layer_number == layer_number_tmp:
			b = b_tmp
		break
	params = [w,b]
	
	#updates, _ = Adam(cost, params, learning_rate)
	updates,_ = RMSProp(cost, params, learning_rate, decay_rate=0.9)
	#updates, _ = Sgd(cost, params, learning_rate)
	train_function = theano.function([x,y], cost, updates=updates,
			allow_input_downcast=True)
	
	test_function = theano.function([x,y], cost,
			allow_input_downcast=True)
	error_function = theano.function([x,y], error,
			allow_input_downcast=True)

	
	x_train, y_train = train
	x_valid, y_valid = valid
	n_train = len(y_train)/batch_size
	stop=False
	init_increment = 5
	increment = init_increment
	error_cost=[]; train_cost=[]
	n_valid = len(y_valid)/batch_size
	for minibatch in range(n_valid):
			x_value = x_valid[minibatch*batch_size:(minibatch+1)*batch_size]
			y_value = y_valid[minibatch*batch_size:(minibatch+1)*batch_size]
			error_cost.append(error_function(x_value, y_value))
			train_cost.append(test_function(x_value, y_value))

	before = np.mean(error_cost)
	if before <=valid_full:
		stop=True
	best_score = np.mean(train_cost)
	best_error = before
	while not stop:
		#print 'LOOP'
		train_cost=[]; error_cost=[]
		for minibatch in range(n_train):
			x_value = x_train[minibatch*batch_size:(minibatch+1)*batch_size]
			y_value = y_train[minibatch*batch_size:(minibatch+1)*batch_size]
			train_function(x_value, y_value)
			if minibatch ==n_train-1:
				if increment !=0:
					for minibatch in range(n_valid):
						x_value = x_valid[minibatch*batch_size:(minibatch+1)*batch_size]
						y_value = y_valid[minibatch*batch_size:(minibatch+1)*batch_size]
						train_cost.append(test_function(x_value, y_value))
						error_cost.append(error_function(x_value, y_value))
					#print np.mean(train_cost)*100
					error = np.mean(error_cost)
					score = np.mean(train_cost)
					if error <= valid_full:
						#print 'A'
						stop=True
						break
					elif score < best_score*0.995:
						best_score = score
						best_error = error
						increment = init_increment
						#print (best_score, np.mean(error_cost)*100)
					else:
						#print 'hihi', increment
						increment-=1
						#print (best_score, score, np.mean(error_cost)*100)
				else:
					stop=True
					break

	# evaluation validation !
	"""
	error_cost=[]
	n_valid = len(y_valid)/batch_size
	for minibatch in range(n_valid):
			x_value = x_valid[minibatch*batch_size:(minibatch+1)*batch_size]
			y_value = y_valid[minibatch*batch_size:(minibatch+1)*batch_size]
			error_cost.append(error_function(x_value, y_value))
	"""
	# new_round with more precision
	if np.mean(error_cost) <=valid_full or learning_rate <=1e-5:
		print (before, best_error*100)
		print '#####'
		return instance
	else:
		return training_committee_member(instance, learning_rate*0.1, train, batch_size, valid, valid_full)
	
def training_committee(committee, learning_rate, train, batch_size, valid, valid_full):
	committee_new = []
	for instance in committee:
		new_instance = training_committee_member(instance, learning_rate, train, batch_size, valid, valid_full)
		committee_new.append(new_instance)
	return committee_new
	
########### TRAINING THE FULL ARCHITECTURE ##########################
def build_training(model=None):
	x = T.tensor4('x')
	y = T.imatrix()
	lr = T.scalar(); mu=T.scalar()
	if model is None:
		model = build_model()

	y_prev = model.apply(x)
	y_softmax =Softmax().apply(y_prev)
	##### prediction #####
	#cost = CategoricalCrossEntropy().apply(y.flatten(), y_prev).mean()
	cost = Softmax().categorical_cross_entropy(y.flatten(), y_prev).mean()
    	error = MisclassificationRate().apply(y.flatten(), y_softmax).mean()
	W, B = get_Params(y_prev)
	params = W + B
	regulizer = sum([w.norm(2) for w in W])
	cost = cost + 0.001*regulizer #+ 0.001*regulizer_conv
	#updates, updates_init = Adam(cost, params, lr)
	updates, updates_init = RMSProp(cost, params, lr)
	#updates, updates_init = Sgd(cost, params, lr)
	train_function = theano.function([lr, x,y], cost, updates=updates,
			allow_input_downcast=True)
	valid_function = theano.function([x,y], cost,
			allow_input_downcast=True)
	test_function = theano.function([x,y], error,
			allow_input_downcast=True)
	reinit = theano.function([], T.zeros((1,)), updates=updates_init)
	"""
	reg_function = theano.function([], T.zeros((1,)), updates=clip(W),
			allow_input_downcast=True)

	observation = theano.function([], [w.norm(2) for w in W])
	"""
	return train_function, valid_function, test_function, model, reinit

