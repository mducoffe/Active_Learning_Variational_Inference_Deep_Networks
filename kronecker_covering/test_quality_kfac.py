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
from kronecker_covering.covering import build_f_u
from kronecker_covering.kronecker_factor import kfac_unlabelled
from kronecker_covering.retrieve_params import mul_theano
import scipy.linalg as lin
from PIL import Image

# REGARDER ORDRE pour kronecker


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
	f = build_f_u(model, filters_info)
	g = mul_theano()
	dico = kfac_unlabelled(x_train, f, g, batch_size=512, dico=None)

	conv_index=0; fully_index=0;
	matrix_diag=[]
	mat_fisher=np.zeros((0,0))
	while "conv_input_"+str(conv_index) in dico:
		A = np.kron(dico['conv_input_'+str(conv_index)],
					dico['conv_output_'+str(conv_index)])
		# add bias
		N = dico['conv_output_'+str(conv_index)].shape[0]
		B = np.zeros((N,N))
		conv_index+=1
		mat_fisher = lin.block_diag(mat_fisher, A, B)
		

	while "fully_input_"+str(fully_index) in dico:
		A = np.kron(dico['fully_input_'+str(fully_index)],
					dico['fully_output_'+str(fully_index)])
		fully_index+=1
		mat_fisher = lin.block_diag(mat_fisher, A)

	return mat_fisher


def training(repo, learning_rate, batch_size, filenames):

	print 'LOAD DATA'
	(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_datasets_mnist(repo, filenames)

	print 'BUILD MODEL'
	train_f, valid_f, test_f, model, fisher, params = build_training()

	x_train = x_train[:1000]; y_train = y_train[:1000]
	#kfac_Fisher = kronecker_Fisher(x_train, y_train, model)
	kfac_Fisher = kronecker_Fisher(x_train, y_train, model)
	import pickle as pkl
	from contextlib import closing
	with closing(open('kfac_fisher', 'wb')) as f:
		pkl.dump(kfac_Fisher, f)
	return
	n_train = len(y_train)/batch_size
	#emp_Fisher = np.zeros_like(kfac_Fisher)
	emp_Fisher = None
	#for minibatch_train in range(n_train):
	for i in range(len(y_train)):
		x_value = x_train[i:i+1]
		y_value = y_train[i:i+1]
		tmp = empiricial_Fisher(params, fisher(x_value, y_value))
		if emp_Fisher is None:
			emp_Fisher = np.zeros_like(tmp)
		emp_Fisher += tmp
	emp_Fisher/=(1.*len(y_train))

	import pickle as pkl
	from contextlib import closing
	with closing(open('true_fisher', 'wb')) as f:
		pkl.dump(emp_Fisher, f)
	return

	#build_obs_Fisher(x_train, y_train, model, batch_size=1)

	n_train = len(y_train)/batch_size
	n_valid = len(y_valid)/batch_size
	n_test = len(y_test)/batch_size
        print n_train, n_valid, n_test
	epochs = 30000
	best_valid = np.inf; best_train = np.inf; best_test=np.inf 
	done = False
	n_train_batches=n_train
	#n_train_batches = n_train
	state_of_train = {}
	state_of_train['TRAIN']=best_train; state_of_train['VALID']=best_valid; state_of_train['TEST']=best_test; 
	print 'TRAINING IN PROGRESS'

	init_increment = 10 #n_train_batches # 20 5 8
	increment = init_increment
	lr = learning_rate

	for epoch in range(epochs):

		try:
			#for minibatch_index in range(n_train_batches):
			minibatch_index=0
			while minibatch_index < n_train_batches:
				x_value = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				value = train_f(lr, x_value, y_value)
				#print value
				minibatch_index+=1
				if np.isnan(value):
					import pdb
					pdb.set_trace()
				if minibatch_index %10==0:
					valid_cost=[]
					for minibatch_index in range(n_valid):
						x_value = x_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
						y_value = y_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
						value = test_f(x_value, y_value)
						valid_cost.append(value)

					# deciding when to stop the training on the sub batch
					valid_result = np.mean(valid_cost)
					#print "ONGOIN valid :"+str(valid_result)
					if valid_result <= best_valid*0.995:
						#print 'OBS', obs()
						#print valid_result*100
						model.save_model() # record the best architecture so to apply active learning on it (overfitting may appear in a few epochs)
	    					best_valid = valid_result
						# compute best_train and best_test
						train_cost=[]
						for minibatch_train in range(n_train_batches):
							x_value = x_train[minibatch_train*batch_size:(minibatch_train+1)*batch_size]
							y_value = y_train[minibatch_train*batch_size:(minibatch_train+1)*batch_size]
							train_cost.append(valid_f(x_value, y_value))
						test_cost=[]
						for minibatch_test in range(n_test):
							x_value = x_test[minibatch_test*batch_size:(minibatch_test+1)*batch_size]
							y_value = y_test[minibatch_test*batch_size:(minibatch_test+1)*batch_size]
							test_cost.append(test_f(x_value, y_value))
						best_train=np.mean(train_cost)
						best_test=np.mean(test_cost)
	    					increment=init_increment
						state_of_train['VALID'] = best_valid
						state_of_train['TEST'] = best_test
						state_of_train['TRAIN'] = best_train
						print 'VALID', best_valid*100
					else:
						increment-=1

					if not done and increment ==0:
						model.load_model()
						increment = init_increment
						if lr <=1e-5:
							done = True
						#train_f, valid_f, test_f, model, reinit = build_training(lr=learning_rate*0.1, model=model)
						lr *=0.1
						minibatch_index=0
					if done and increment==0:

						# keep the best set of params found during training
						model.load_model()
						increment = init_increment
						done = False
						print "RATIO :"+str(1.*n_train_batches/n_train*100)
						print state_of_train
						print (best_valid, best_test)
						# build the hessian
						# record it

						
						"""
						# 1) KFAC fisher matrix
						kfac_Fisher = kronecker_Fisher(x_train, y_train, model)
						# 2) empirical fisher matrix
						emp_Fisher = np.zeros_like(kfac_Fisher)
						for minibatch_train in range(n_train_batches):
							x_value = x_train[minibatch_train*batch_size:(minibatch_train+1)*batch_size]
							y_value = y_train[minibatch_train*batch_size:(minibatch_train+1)*batch_size]
							tmp = empiricial_Fisher(params, fisher(x_value, y_value))
							emp_Fisher += tmp
						emp_Fisher/=(1.*n_train_batches)
						# 3) diagonal fisher matrix
						diag_Fisher = np.diag([emp_Fisher[i,i] for i in range(len(emp_Fisher))])

						# normalization between 0, 255
						max_value = np.max(emp_Fisher)
						min_value = np.min(emp_Fisher)

						kfac_Fisher = (((kfac_Fisher - min_value)/(max_value - min_value))*255.9).astype(np.uint8)
						kfac_Fisher = np.clip(kfac_Fisher, 0, 255)
						
						emp_Fisher = (((emp_Fisher - min_value)/(max_value - min_value))*255.9).astype(np.uint8)
						emp_Fisher = np.clip(emp_Fisher, 0, 255)

						diag_Fisher = (((diag_Fisher - min_value)/(max_value - min_value))*255.9).astype(np.uint8)
						diag_Fisher = np.clip(diag_Fisher, 0, 255)

						img_kfac = Image.fromarray(kfac_Fisher)#.convert('LA')
						img_kfac.save('kfac_fisher.png')

						img_emp = Image.fromarray(emp_Fisher)#.convert('LA')
						img_emp.save('emp_fisher.png')

						img_diag = Image.fromarray(diag_Fisher)#.convert('LA')
						img_diag.save('diag_fisher.png')
						"""
						return

		except KeyboardInterrupt:
			# ask confirmation if you want to check state of training or really quit
			print 'BEST STATE OF TRAINING ACHIEVED'
			print "RATIO :"+str(1.*n_train_batches/n_train*100)
			print "TRAIN : "+str(state_of_train['TRAIN']*100)
			print "VALID : "+str(state_of_train['VALID']*100)
			print "TEST : "+str(state_of_train['TEST']*100)
			import pdb
			pdb.set_trace()


if __name__=="__main__":
	print 'kikou'
	repo="/home/ducoffe/Documents/Code/datasets/mnist"
	filenames="mnist.pkl.gz"
	learning_rate = 1e-3
	batch_size = 64
	training(repo, learning_rate, batch_size,filenames)

