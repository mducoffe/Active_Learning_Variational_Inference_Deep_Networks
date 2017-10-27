import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import (Initializable, Feedforward, Sequence,
                            Rectifier, Tanh, Logistic, Identity, MLP, Linear,
                            Softmax, BatchNormalization)
from blocks.roles import WEIGHT, BIAS, INPUT, OUTPUT
from blocks.bricks.conv import (ConvolutionalSequence, Flattener, MaxPooling, Convolutional)
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
import scipy.linalg as lin
from blocks.initialization import Constant, Uniform, IsotropicGaussian

def expansion_op(A, shape, delta):
	#shape (M, J, X, Y)
	(M, J, X, Y) = shape
	d_x = delta[0]/2; d_y = delta[1]/2
	E_A = np.zeros((M, X -2*d_x, Y -2*d_y, J, 2*d_x+1, 2*d_y+1), dtype=A.dtype)

	for m in range(M):
		for j in range(J):
			for n_x in range(d_x, X-d_x):
				for n_y in range(d_y, Y-d_y):
					E_A[m, n_x -d_x, n_y -d_y,j] = np.flipud(np.fliplr(A[m,j, n_x-d_x:n_x+d_x+1, n_y-d_y:n_y+d_y+1]))

	E_A = E_A.reshape((M, (X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1)))
	return E_A
	Id = np.diag([1]*(X-2*d_x)*(Y-2*d_y))
	E_B = np.ones((M, (X-2*d_x)*(Y-2*d_y), J*(2*d_x+1)*(2*d_y+1) + (X-2*d_x)*(Y-2*d_y)))
	for m in range(M):
		E_B[m] = np.concatenate([E_A[m], Id], axis=1)
	return E_B

def test_convolutional_layer():
	batch_size=2
	x = T.tensor4();
	y = T.ivector()
	V = 200
	layer_conv = Convolutional(filter_size=(5,5),num_filters=V,
				name="toto",
				weights_init=IsotropicGaussian(0.01),
				biases_init=Constant(0.0))
	# try with no bias
	activation = Rectifier()
	pool = MaxPooling(pooling_size=(2,2))

	convnet = ConvolutionalSequence([layer_conv, activation, pool], num_channels=15,
					image_size=(10,10),
					name="conv_section")
	convnet.push_allocation_config()
	convnet.initialize()
	output=convnet.apply(x)
	batch_size=output.shape[0]
	output_dim=np.prod(convnet.get_dim('output'))
	result_conv = output.reshape((batch_size, output_dim))
	mlp=MLP(activations=[Rectifier().apply], dims=[output_dim, 10],
				weights_init=IsotropicGaussian(0.01),
				biases_init=Constant(0.0))
	mlp.initialize()
	output=mlp.apply(result_conv)
	cost = T.mean(Softmax().categorical_cross_entropy(y.flatten(), output))
	cg = ComputationGraph(cost)
	W = VariableFilter(roles=[WEIGHT])(cg.variables)
	B = VariableFilter(roles=[BIAS])(cg.variables)
	W = W[-1]; b = B[-1]
	
	print W.shape.eval()
	print b.shape.eval()
	import pdb
	pdb.set_trace()
	inputs_conv = VariableFilter(roles=[INPUT], bricks=[Convolutional])(cg)
	outputs_conv = VariableFilter(roles=[OUTPUT], bricks=[Convolutional])(cg)
	var_input=inputs_conv[0]
	var_output=outputs_conv[0]
	
	[d_W,d_S,d_b] = T.grad(cost, [W, var_output, b])

	import pdb
	pdb.set_trace()
	w_shape = W.shape.eval()
	d_W = d_W.reshape((w_shape[0], w_shape[1]*w_shape[2]*w_shape[3]))

	d_b = T.zeros((w_shape[0],6*6))
	#d_b = d_b.reshape((w_shape[0], 8*8))
	d_p = T.concatenate([d_W, d_b], axis=1)
	d_S = d_S.dimshuffle((1, 0, 2, 3)).reshape((w_shape[0], batch_size, 6*6)).reshape((w_shape[0], batch_size*6*6))
	#d_S = d_S.reshape((2,200, 64))
	#x_value=1e3*np.random.ranf((1,15,10,10))
	x_value = 1e3*np.random.ranf((2,15, 10, 10))
	f = theano.function([x,y], [var_input, d_S, d_W], allow_input_downcast=True, on_unused_input='ignore')
	A, B, C= f(x_value, [5, 5])
	print np.mean(B)
	return
	
	E_A = expansion_op(A, (2, 15, 10, 10), (5,5))
	print E_A.shape
	E_A = E_A.reshape((2*36, C.shape[1]))
	print E_A.shape
	tmp = C - np.dot(B, E_A)
	print lin.norm(tmp, 'fro')

def test_fully_layer():
	batch_size=2
	x = T.tensor4();
	y = T.ivector()
	V = 200
	layer_conv = Convolutional(filter_size=(5,5),num_filters=V,
				name="toto",
				weights_init=IsotropicGaussian(0.01),
				biases_init=Constant(0.0))
	# try with no bias
	activation = Rectifier()
	pool = MaxPooling(pooling_size=(2,2))

	convnet = ConvolutionalSequence([layer_conv, activation, pool], num_channels=15,
					image_size=(10,10),
					name="conv_section")
	convnet.push_allocation_config()
	convnet.initialize()
	output=convnet.apply(x)
	batch_size=output.shape[0]
	output_dim=np.prod(convnet.get_dim('output'))
	result_conv = output.reshape((batch_size, output_dim))
	mlp=MLP(activations=[Rectifier().apply], dims=[output_dim, 10],
				weights_init=IsotropicGaussian(0.01),
				biases_init=Constant(0.0))
	mlp.initialize()
	output=mlp.apply(result_conv)
	cost = T.mean(Softmax().categorical_cross_entropy(y.flatten(), output))
	cg = ComputationGraph(cost)
	W = VariableFilter(roles=[WEIGHT])(cg.variables)
	B = VariableFilter(roles=[BIAS])(cg.variables)
	W = W[0]; b = B[0]

	inputs_fully = VariableFilter(roles=[INPUT], bricks=[Linear])(cg)
	outputs_fully = VariableFilter(roles=[OUTPUT], bricks=[Linear])(cg)
	var_input=inputs_fully[0]
	var_output=outputs_fully[0]
	
	[d_W,d_S,d_b] = T.grad(cost, [W, var_output, b])

	d_b = d_b.dimshuffle(('x',0))
	d_p = T.concatenate([d_W, d_b], axis=0)
	x_value = 1e3*np.random.ranf((2,15, 10, 10))
	f = theano.function([x,y], [var_input, d_S, d_p], allow_input_downcast=True, on_unused_input='ignore')
	A, B, C= f(x_value, [5, 0])
	A = np.concatenate([A, np.ones((2,1))], axis=1)
	print 'A', A.shape
	print 'B', B.shape
	print 'C', C.shape

	print lin.norm(C - np.dot(np.transpose(A), B), 'fro')

	return
	
	"""
	print E_A.shape
	E_A = E_A.reshape((2*36, C.shape[1]))
	print E_A.shape
	tmp = C - np.dot(B, E_A)
	"""



if __name__=="__main__":
	test_convolutional_layer()
		


