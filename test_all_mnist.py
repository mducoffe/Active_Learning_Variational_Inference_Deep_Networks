# test model for mnist ##
# @date 8/05/2016
# @author Melanie Ducoffe
#########################

from load_data import load_datasets_mnist
from training_mnist import build_training, training_committee
import numpy as np
import theano
import theano.tensor as T
from active_function import active_selection
from contextlib import closing
import os


def swap_data(x, y, i, j):
	tmp_x = x[i]
	tmp_y = y[i]
	x[i] = x[j]; y[i] = y[j];
	x[j]=tmp_x; y[j]=tmp_y
	return x, y

def permut_data(x,y):
	i_tab = range(len(y))
	j_tab = np.random.permutation(len(y))
	x_ = np.zeros_like(x)
	y_ = np.zeros_like(y)
	for i, j in zip(i_tab, j_tab):
		x_[i] = x[j]
		y_[i] = y[j]
	return x_,y_

def training(repo, learning_rate, batch_size, filenames, percentage=1):

	momentum=0.5
	print 'LOAD DATA'
	(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_datasets_mnist(repo, filenames)
	x_train, y_train = permut_data(x_train, y_train)

	print 'BUILD MODEL'
	train_f, valid_f, test_f, model, reinit = build_training()

	n_train = len(y_train)/batch_size
	n_train = (int) (n_train*percentage)
	n_valid = len(y_valid)/batch_size
	n_test = len(y_test)/batch_size
	init_lr_decay = 6
	done=False
	increment = init_lr_decay
	epochs = 2000
	best_valid = np.inf; best_train = np.inf; best_test=np.inf
	state_of_train = {}
	state_of_train['TRAIN']=best_train; state_of_train['VALID']=best_valid; state_of_train['TEST']=best_test; 
	print 'TRAINING IN PROGRESS'
	for epoch in range(epochs):
			if epoch +1%5==0:
				learning_rate=learning_rate*(1 - 0.2/epoch)
			for minibatch_index in range(n_train):
				x_value = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				value = train_f(learning_rate, x_value, y_value)
				if np.isnan(value):
					import pdb
					pdb.set_trace()
			valid_cost=[]
			for minibatch_index in range(n_valid):
				x_value = x_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				y_value = y_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				value = test_f(x_value, y_value)
				valid_cost.append(value)

			# deciding when to stop the training on the sub batch
	    		valid_result = np.mean(valid_cost)
	    		if valid_result <= best_valid*0.95:
				increment = init_lr_decay
	    			best_valid = valid_result
				# compute best_train and best_test
				train_cost=[]
				for minibatch_train in range(n_train):
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
				#print "TRAIN : "+str(best_train)
				#print "VALID : "+str(best_valid*100)
				#print "TEST : "+str(best_test*100)
			else:
				increment -=1


			if not done and increment ==0:
				learning_rate/=2.
				#train_f, valid_f, test_f, model, reinit = build_training(lr=learning_rate*0.1, model=model)
				increment = init_lr_decay; done=True
			if increment ==0:
				print 'END OF TRAINING'
				print percentage*100
				print "TRAIN : "+str(best_train)
				print "VALID : "+str(best_valid*100)
				print "TEST : "+str(best_test*100)
				return

if __name__=="__main__":

	import sys
	percentage=1
	if len(sys.argv)>=2:
		percentage=(float) (sys.argv[1])

	repo="/home/mducoffe/Documents/Code/datasets/mnist"
	filenames="mnist.pkl.gz"
	learning_rate = 1e-3
	batch_size = 32
	training(repo, learning_rate, batch_size,filenames, percentage)
