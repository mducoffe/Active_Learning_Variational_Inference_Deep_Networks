#### COMPARISON WITH OBSERVED FISHER MATRIX AND THE KFAC IMPLEMENTATION #####
# @author Melanie Ducoffe
# @date 18/08/2016
#############################################################################
import sys
sys.path.append('..')
import theano
from load_data import load_datasets_mnist
from training_mnist import build_training, get_Params
import numpy as np
import theano.tensor as T
from blocks.bricks.cost import MisclassificationRate, CategoricalCrossEntropy
from blocks.bricks import Softmax

from PIL import Image
from kronecker_covering.covering import build_f_l
from kronecker_covering.kronecker_factor import kfac_labelled
from kronecker_covering.retrieve_params import mul_theano

from blocks.bricks import Softmax
from blocks.graph import ComputationGraph
from blocks.bricks.conv import Convolutional
from blocks.bricks import Linear
from blocks.roles import WEIGHT, BIAS, INPUT, OUTPUT
from blocks.filter import VariableFilter
from collections import OrderedDict


#def empirical_Giser(x_train, y_train, 
def empiricial_Fisher(params, dico_grad):
	fisher_row=[]
	for p_0 in params:
		g_0 = dico_grad[p_0.name]
		g_0 = g_0
		row_p=[]
		for p_1 in params:
			g_1 = dico_grad[p_1.name]
			g_1 = np.transpose(g_1)
			row_p.append(np.dot(g_0, g_1))
		fisher_row.append(np.concatenate(row_p, axis=1))
	return np. concatenate(fisher_row, axis=0)

def kronecker_Fisher(x_train, y_train, model, batch_size=512):
	filters_info = model.filters_info
	f = build_f_l(model, filters_info)
	g = mul_theano()
	return kfac_labelled(x_train, y_train, f, g, batch_size=512, dico=None)


def training(repo, learning_rate, batch_size, filenames):

	print 'LOAD DATA'
	(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_datasets_mnist(repo, filenames)

	print 'BUILD MODEL'
	train_f, valid_f, test_f, model, fisher, params = build_training()
	x_train = x_train[:1000]; y_train = y_train[:1000]

	x = T.tensor4(); y = T.imatrix();
	output = model.apply(x)
	output = output.reshape((x.shape[0], model.get_dim('output'))) #TO DO : get_dim('name') for Architecture
	cost = Softmax().categorical_cross_entropy(y.flatten(), output).mean()
	cg = ComputationGraph(cost)

	inputs_conv = VariableFilter(roles=[INPUT], bricks=[Convolutional])(cg)
	outputs_conv = VariableFilter(roles=[OUTPUT], bricks=[Convolutional])(cg)
	inputs_fully = VariableFilter(roles=[INPUT], bricks=[Linear])(cg)
	outputs_fully = VariableFilter(roles=[OUTPUT], bricks=[Linear])(cg)
	dico = OrderedDict([('conv_output', outputs_conv[0])])
	[grad_s] = T.grad(cost, outputs_conv)
	dico['conv_output']=grad_s
	
	f=theano.function([x,y], grad_s, allow_input_downcast=True, on_unused_input='ignore')
	print np.mean(f(x_train[:10], y_train[:10]))

if __name__=="__main__":
	print 'kikou'
	repo="/home/ducoffe/Documents/Code/datasets/mnist"
	filenames="mnist.pkl.gz"
	learning_rate = 1e-3
	batch_size = 64
	training(repo, learning_rate, batch_size,filenames)

