### error loss on a multivariate gaussian distribution 
import numpy as np
import theano
import theano.tensor as T
from blocks.bricks import Softmax
from training_mnist import get_Params
from blocks.graph import ComputationGraph
from blocks.filter import VariableFilter
from blocks.roles import WEIGHT
from blocks.bricks.conv import Convolutional
from blocks.bricks import Linear
from retrieve_params import dot_theano

"""
inserer loss dans fonction greedy
apporter optim gpu
corriger trace avec KL
"""

def generate_sample(N):
	return np.array([np.random.ranf(1)[0] for n in range(N)])
	#return np.random.multivariate_normal(mean, covariance, 1)[0]


def generate_weights(means, kron_covariances, tab_equiv, names, shapes_W, multiply=None):
	params = []
	for mean, name in zip(means, names):
		cov_input = kron_covariances[tab_equiv[name][0]]
		cov_output = kron_covariances[tab_equiv[name][1]]
		params.append(build_sample_layer(mean, cov_input, cov_output, multiply)) # TO DO optimisation with gpu multiplication

	W_value=[]; B_value=[]
	for param, shape in zip(params, shapes_W):
		if len(shape)==4:
			W_value.append(param)
			B_value.append(np.zeros((shape[0],1)))
		if len(shape)==2:
			shape_b = shape[1]
			W_value.append(param[:len(param)-shape_b])
			B_value.append(param[len(param)-shape_b:])
	W_value = [w.astype(np.float32) for w in W_value]
	B_value = [b.astype(np.float32) for b in B_value]
	return W_value, B_value

def build_mean(model):
	means=[]
	x = T.tensor4('x'); y = T.imatrix()
	y_prev = model.apply(x)
	W, B = get_Params(y_prev)
	shapes = [w.ndim for w in W]
	for w, b, dim in zip(W,B, shapes):
		if dim==4:
			mean=w.get_value().flatten()
		if dim==2:
			mean=np.concatenate([w.get_value().flatten(), b.get_value().flatten()], axis=0)
		means.append(mean)

	return means


def build_tab_equiv(model):
	x = T.tensor4('x'); y = T.imatrix()
	y_prev = model.apply(x)
	cg = ComputationGraph(T.sum(y_prev))

	weight_fully = VariableFilter(roles=[WEIGHT], bricks=[Linear])(cg)
	weight_conv = VariableFilter(roles=[WEIGHT], bricks=[Convolutional])(cg)

	dico={}
	index=0
	for w_fully in weight_fully[::-1]:
		dico[w_fully.name] = ['fully_input_'+str(index), 'fully_output_'+str(index)]
		index+=1
	
	index=0
	for w_conv in weight_conv[::-1]:
		dico[w_conv.name] = ['conv_input_'+str(index), 'conv_output_'+str(index)]
		index+=1

	return dico


def build_covariance(dico, tab_equiv, multiply=None):
	# tab equiv : table of equivalency reguarding the names used
	covariances=[]
	for name in tab_equiv.keys():
		input_name = tab_equiv[name][0];
		output_name = tab_equiv[name][1];
		if input_name[:4]=='conv':
			# convolution
			build_sample_layer(dico[input_name], dico[output_name], multiply)
		else:
			print 'not conv', input_name
		"""
		if kronecker is None:
			covariances.append(np.kron(dico[input_name], dico[output_name]))
		else:
			print name, dico[input_name].shape, dico[output_name].shape
			covariances.append(kronecker(dico[input_name], dico[output_name]))
			# check for bias on fully !!!!!
		"""
	return covariances

def build_sample_layer(mean, cov_input, cov_output, multiply=None):

	A = cov_input.shape[0]; B = cov_output.shape[0]
	x_s = [ generate_sample(B) for n in range(A)]
	X = np.array(x_s) # [X_1, X_2, ..., X_n] #shape(A,B)
	X = X.transpose((1,0))

	if multiply is None:
		multiply = dot_theano(); # multiplication with gpu

	# (B kro A)vec X=C equiv AXB.T = C
	Y = multiply(multiply(cov_output, X), cov_input.transpose((1,0)))
	# vectorization not equivalent to flatten in numpy !!!
	Y = Y.reshape((-1,1), order="F")
	return Y + mean[:,None]


def put_labels(model, data_labelled, data_unlabelled, f_predict=None, f_loss=None):
	x = T.tensor4('x'); y = T.imatrix()
	y_prev = model.apply(x)

	if f_predict is None:
		# define prediction function
		y_softmax = Softmax().apply(y_prev)
		prediction = T.argmax(y_softmax, axis=1)
		f_predict = theano.function([x], prediction, allow_input_downcast=True)
	if f_loss is None:
		cost = Softmax().categorical_cross_entropy(y.flatten(), y_prev).mean()
		f_loss = theano.function([x,y], cost, allow_input_downcast=True)


	# now proceed to the mean loss
	batch_size = 64
	x_train_L, y_train_L = data_labelled
	y_train_U =[]; x_train_U = data_unlabelled
	n_train_U = len(x_train_U)/batch_size
	# pay attention to the shape of y_train !!!!!!!!!!
	for index in range(n_train_U):
		y_train_batch = f_predict(x_train_U[batch_size*index:(index+1)*batch_size])
		y_train_U.append(y_train_batch) # what is the type of predict
	if n_train_U*batch_size < len(x_train_U):
		x_batch = x_train_U[n_train_U*batch_size:]
		y_train_U.append(f_predict(x_batch))


	y_train_U = np.concatenate(y_train_U, axis=0)[:,None]
	assert len(y_train_U)==len(x_train_U), "problem : length does not match for unlabelled data"


	y_train = np.concatenate([y_train_L, y_train_U], axis=0)
	x_train = np.concatenate([x_train_L, x_train_U], axis=0)
	return (x_train, y_train), f_loss

def error_loss(data, means, kron_covariances, tab_equiv, model, nb_trial=5, f_loss=None, multiply=None):
	loss = 0;
	for i in range(nb_trial):
		loss_sample= error_loss_priv(data, means, kron_covariances, tab_equiv, model, f_loss, multiply)	
		loss += loss_sample

	return loss/nb_trial;	

def error_loss_priv(data, means, kron_covariances, tab_equiv, model,f_loss, multiply=None):
	x = T.tensor4('x'); y = T.imatrix()
	y_prev = model.apply(x)

	# copy the weights in the model
	W, B = get_Params(y_prev)
	# faire matcher les noms

	W_value, B_value = generate_weights(means, kron_covariances, tab_equiv, [w.name for w in W], [w.shape.eval() for w in W], multiply) # TO DO

	for w, w_value in zip(W, W_value):
		shape_w = w.shape.eval()
		w.set_value(w_value.reshape(shape_w))

	for b, b_value in zip(B, B_value):
		shape_b = b.shape.eval()
		b.set_value(b_value.reshape(shape_b))

	# now proceed to the mean loss
	batch_size = 64
	x_train, y_train = data
	n_train = len(y_train)/batch_size
	mean_loss=[]

	for index in range(n_train):
		x_batch = x_train[batch_size*index:(index+1)*batch_size]
		y_batch = y_train[batch_size*index:(index+1)*batch_size]

		mean_loss.append(f_loss(x_batch, y_batch))

	if n_train*batch_size < len(y_train):
		x_batch = x_train[n_train*batch_size:]
		y_batch = y_train[n_train*batch_size:]
		mean_loss.append(f_loss(x_batch, y_batch))

	loss = np.mean(mean_loss)
	return loss
