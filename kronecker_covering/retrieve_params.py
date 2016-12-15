#### RETRIEVE PARAMS #####
# @date 26/04/2016
# @author Melanie Ducoffe

import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict  # this can be moved to the top with the other imports
from blocks.bricks import Softmax
from blocks.graph import ComputationGraph
from blocks.bricks.conv import Convolutional
from blocks.bricks import Linear
from blocks.roles import WEIGHT, BIAS, INPUT, OUTPUT
from blocks.filter import VariableFilter

def dot_theano():
	A = T.matrix()
	B = T.matrix()
	def f(A,B):
		return T.dot(A,B)
	g = theano.function([A,B], f(A,B), allow_input_downcast=True)
	return g


def mul_theano():
	M = T.matrix()
	def f(M):
		return T.dot(M.transpose(), M)

	g = theano.function([M], f(M), allow_input_downcast=True)
	return g

def compute_coeff():
	A = T.matrix(); B = T.matrix()
	det = T.nlinalg.Det()
	def f(A,B):
		C = T.dot(A,B)
		D = T.nlinalg.trace(C)
		#shape = A.shape[0]
		#E = T.pow(det(C), shape)
		return D

	g = theano.function([A,B], f(A,B), allow_input_downcast=True)
	return g;
"""
def mul_theano():
	M = T.tensor3()
	index = T.scalar()
	def f(index, M):
		return T.dot(M[index].transpose(), M[index])
	output, updates = theano.scan(fn=f, outputs_info=None,
				      sequences=[T.arange(M.shape[0])],
				      non_sequences=[M])

	g = theano.function([M], output, allow_input_downcast=True)
	def function(matrices):
		if matrices.ndim==2:
			matrices = matrices.reshape((matrices.shape[0], 1, matrices.shape[1]))
		""
		if matrices.shape[2] > 2000:
			result = [ g(matrices[i:i+1]) for i in range(matrices.shape[0])]
		""
		return g(matrices)
	return function
"""

# TO DO integrer 1/sqrt(tau)
def expansion_op(A, delta):
	# shape = (M, J, X, Y)
	(M, J, X, Y) = A.shape
	d_x = delta[0]/2; d_y = delta[1]/2
	E_A = np.zeros((M,X -2*d_x, Y - 2*d_y, J, 2*d_x+1, 2*d_y+1), dtype=A.dtype)
	for m in range(M):
		for j in range(J):
			for n_x in range(d_x, X -d_x):
				for n_y in range(d_y, Y-d_y):
					E_A[m,n_x-d_x, n_y-d_y, j] = np.flipud(np.fliplr(A[m,j, n_x -d_x:n_x+d_x+1, n_y-d_y:n_y+d_y+1]))

	coeff = np.sqrt(1./((X-2*d_x)*(Y-2*d_y))) # 1/sqrt(tau)
	E_A = E_A.reshape((M*(X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1)))
	return E_A
	# including bias has no sense, it changes the dimensions !!!
	"""
	Id = np.diag([1]*((X-2*d_x)*(Y-2*d_y)))
	E_B = np.zeros((M,(X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1) + (X-2*d_x)*(Y-2*d_y)))
	for m in range(M):
		E_B[m] = np.concatenate([E_A[m], Id], axis=1)
	#import pdb
	#pdb.set_trace()
	#E_A = np.concatenate([E_A, Id], axis=1)
	return E_B.reshape((E_B.shape[0]*E_B.shape[1], E_B.shape[2]))
	return E_B
	"""


def expansion_softmax(prob):
	shape = prob.shape #batch_size, nb_classe
	# add bias TOCHECK
	I = np.ones((shape[0], shape[1], shape[1]))
	max_prob = np.max(prob, axis=1)
	for m in range(shape[0]):
		for i in range(shape[1]):
			I[m, i:(i+1), i:(i+1)] *= -prob[m,i]*(1 -prob[m,i])
			for j in range(i, shape[1]):
				I[m, i:(i+1), j:(j+1)] *= prob[m,i]*prob[m,j]
				I[m, j:(j+1), i:(i+1)] *= prob[m,i]*prob[m,j]
		I[m]*=max_prob[m]
	return I
	

def preprocessing_blocks(dico, filters):
	keys = dico.keys()
	i = 0
	while "conv_input_"+str(i) in keys:
		var_input = dico["conv_input_"+str(i)]
		dico["conv_input_"+str(i)] = expansion_op(var_input, filters[i])
		dico["conv_output_"+str(i)] = np.transpose(dico["conv_output_"+str(i)], (1,0, 2,3))
		shape = dico["conv_output_"+str(i)].shape
		dico["conv_output_"+str(i)] = dico["conv_output_"+str(i)].reshape((shape[0], shape[1], shape[2]*shape[3]))
		dico["conv_output_"+str(i)] = dico["conv_output_"+str(i)].reshape((shape[0], shape[1]*shape[2]*shape[3]))
		dico["conv_output_"+str(i)] = np.transpose(dico["conv_output_"+str(i)])
		i+=1
	
	# logistic
	if "logistic_output" in keys:
		dico["logistic_output"] = expansion_softmax(dico["logistic_output"])
		shape = dico["logistic_input"].shape
		tmp =  np.zeros((shape[0], shape[1]+1, shape[1]+1))
		for m in range(shape[0]):
			data = np.concatenate([dico["logistic_input"][m], np.ones((1,))], axis=0)
			data = data.reshape((data.shape[0], 1))
			tmp[m] = np.dot(data, np.transpose(data))
		dico["logistic_input"] = tmp
	return dico

def build_dictionnary(cost):
	cg = ComputationGraph(cost)
	
	inputs_conv = VariableFilter(roles=[INPUT], bricks=[Convolutional])(cg)
	outputs_conv = VariableFilter(roles=[OUTPUT], bricks=[Convolutional])(cg)
	inputs_fully = VariableFilter(roles=[INPUT], bricks=[Linear])(cg)
	outputs_fully = VariableFilter(roles=[OUTPUT], bricks=[Linear])(cg)
	grad_conv = T.grad(cost,outputs_conv)
	grad_fully = T.grad(cost, outputs_fully)

	items=[]
	for i, var_in, grad_out in zip(range(len(inputs_conv)), inputs_conv, grad_conv):
		items.append(('conv_input_'+str(i), var_in))
		items.append(('conv_output_'+str(i), grad_out))

	for i, var_in, grad_out in zip(range(len(inputs_fully)), inputs_fully, grad_fully):
		var_input = T.concatenate([var_in, T.ones((var_in.shape[0], 1))], axis=1)
		items.append(('fully_input_'+str(i), var_input))
		items.append(('fully_output_'+str(i), grad_out))

	dico = OrderedDict(items)

	return dico


def build_intermediate_var_labelled(model):
	# build cost function and computational
	x = T.tensor4(); y = T.imatrix();
	output = model.apply(x)
	output = output.reshape((x.shape[0], model.get_dim('output'))) #TO DO : get_dim('name') for Architecture
	cost = Softmax().categorical_cross_entropy(y.flatten(), output).mean()
	return theano.function([x,y], build_dictionnary(cost), allow_input_downcast=True, on_unused_input='ignore')


def build_intermediate_var_unlabelled(model):
	# build cost function and computational
	x = T.tensor4();
	output = model.apply(x)
	output = output.reshape((x.shape[0], model.get_dim('output')))
	labels = T.argmax(output, axis=1).reshape((x.shape[0], 1))
	cost = Softmax().categorical_cross_entropy(labels.flatten(), output).mean()
	return theano.function([x], build_dictionnary(cost), allow_input_downcast=True, on_unused_input='ignore')
