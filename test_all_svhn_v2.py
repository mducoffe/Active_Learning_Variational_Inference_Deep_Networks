####### TRAINING CIFAR-10 CONVNET ##############
# @author Melanie Ducoffe #
# @date 10/02/2016 #
####################
from load_data import build_datasets, load_datasets
from training_svhn import build_training
import numpy as np
import pickle
from contextlib import closing
import os

def shuffle_data(x,y):
	x_ = np.zeros_like(x)
	y_ = np.zeros_like(y)
	n = len(y)
	index = np.random.permutation(n)
	for i,j in zip(range(n), index):
		x_[i]=x[j]
		y_[i]=y[j]
	return x_, y_


def training(repo, learning_rate, batch_size, proportion, pickle_f="svhn_zca.pkl"):
	#train, valid, test = build_datasets(repo)

	repo="/home/ducoffe/Documents/Code/datasets/svhn"
	#build_datasets(repo)
	import h5py
	with closing(h5py.File(os.path.join(repo, pickle_f), mode='r')) as f:
		x = np.array(f['features'])
		y = np.array(f['targets'])
	
	x,y = shuffle_data(x, y)

	n_train = (int)(len(y)*0.8)
	n_valid = (int)(len(y)*0.9)
	x_train, y_train = x[:n_train], y[:n_train]
	x_valid, y_valid = x[n_train:n_valid], y[n_train:n_valid]
	x_test, y_test = x[n_valid:], y[n_valid:]
	
	#train_f, valid_f, test_f, model, reinit = build_training(lr=learning_rate)
	train_f, valid_f, test_f, model, obs = build_training(lr=learning_rate)
	n_train = len(y_train)/batch_size
	n_valid = len(y_valid)/batch_size
	n_test = len(y_test)/batch_size

	mean = np.mean(x_train, axis=0)
	std = np.std(x_train, axis=0)

	x_train = (x_train -mean)/std
	x_valid = (x_valid -mean)/std
	x_test = (x_test -mean)/std

	# cut x_train
	print n_train
	n_train = (int)(1.*proportion/100.*n_train)
	print n_train
	x_train = x_train[:n_train*batch_size]; y_train =y_train[:n_train*batch_size]

	epochs = 20
	best_cost = np.inf
	init_increment = 10
	increment = init_increment
	for epoch in range(epochs):
		for minibatch_index in range(n_train):
			x_value = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size].reshape((batch_size, 3, 32, 32))
			y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size].reshape((batch_size, 1))
			value = train_f(x_value, y_value)
			if minibatch_index %50==0:
				valid_cost=[]
				for minibatch_valid in range(n_valid):
					x_value = x_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size].reshape((batch_size, 3, 32, 32))
					y_value = y_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size].reshape((batch_size, 1))
					valid_cost.append(test_f(x_value, y_value))
				valid_score = np.mean(valid_cost)*100
				#print 'ONGOIN :'+str(valid_score)
				if increment !=0:
					#print 'coco'
					if valid_score < best_cost*0.995:
						increment = init_increment
						best_cost = valid_score
						valid_error = []
						for minibatch_valid in range(n_valid):
							x_value = x_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size].reshape((batch_size, 3, 32, 32))
							y_value = y_valid[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size].reshape((batch_size, 1))
							valid_error.append(test_f(x_value, y_value))
						test_error =[]
						for minibatch_valid in range(n_test):
							x_value = x_test[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size].reshape((batch_size, 3, 32, 32))
							y_value = y_test[minibatch_valid*batch_size:(minibatch_valid+1)*batch_size].reshape((batch_size, 1))
							test_error.append(test_f(x_value, y_value))
						print "VALID : "+str(np.mean(valid_error)*100)
						print "TEST : "+str(np.mean(test_error)*100)
					else:
						increment -=1
				else:
					print 'START AGAIN'
					train_f, valid_f, test_f,_, obs = build_training(lr=learning_rate*0.1, model=model)
					increment = init_increment
					break
				#print obs()

if __name__=="__main__":
	import sys
	proportion=100
	if len(sys.argv) >=2:
		proportion = (int) (sys.argv[-1])
	
	repo="/home/ubuntu/Documents/Code/dataset/svhn_tar_gz"
	learning_rate = 2*1e-3
	batch_size = 64
	training(repo, learning_rate, batch_size, proportion)
	
