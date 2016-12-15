####### TRAINING CIFAR-10 CONVNET ##############
# @author Melanie Ducoffe #
# @date 10/02/2016 #
####################
from load_data import load_datasets_mnist
from training_mnist import build_training, training_committee
import numpy as np
import theano
import theano.tensor as T
from active_function import active_selection
from contextlib import closing
import os

def record_state(n_train_batches, n_train, best_train, best_valid, best_test, repo=None, filename=None):
	ratio = "RATIO :"+str(1.*n_train_batches/n_train*100)
	train = "TRAIN : "+str(best_train*100)
	valid = "VALID : "+str(best_valid*100)
	test = "TEST : "+str(best_test*100)
	if filename is None:
		print ratio
		print train
		print valid
		print test
	else:
		with closing(open(os.path.join(repo, filename), 'a')) as f:
			f.write(ratio+"\n"+train+"\n"+valid+"\n"+test+"\n")

"""
3 options possible :
- 'rand' for random selection
- 'uncertainty_s' for uncertain sampling
- 'qbc' for query by committee
"""

def permut_data(x,y):
	i_tab = range(len(y))
	j_tab = np.random.permutation(len(y))
	x_ = np.zeros_like(x)
	y_ = np.zeros_like(y)
	for i, j in zip(i_tab, j_tab):
		x_[i] = x[j]
		y_[i] = y[j]
	return x_,y_
 
def training(repo, learning_rate, batch_size, filenames, option, record_repo=None, record_filename=None, nb_to_query=1):
	lr_init = learning_rate; epoch_tmp=0
	print 'LOAD DATA'
	(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = load_datasets_mnist(repo, filenames)
	x_train, y_train = permut_data(x_train, y_train)

	print 'BUILD MODEL'
	train_f, valid_f, test_f, model, reinit = build_training()

	n_train = len(y_train)/batch_size
	n_valid = len(y_valid)/batch_size
	n_test = len(y_test)/batch_size

	epochs = 300000
	best_valid = np.inf; best_train = np.inf; best_test=np.inf
	init_increment = 5 # 20 5 8 
	done = False
	increment = init_increment
	#in_train_batches=int (n_train*0.1/100.)
        n_train_batches=1
	state_of_train = {}
	state_of_train['TRAIN']=best_train; state_of_train['VALID']=best_valid; state_of_train['TEST']=best_test; 
	print 'TRAINING IN PROGRESS'

	for epoch in range(epochs):

		try:
			if epoch_tmp +1%5==0:
				learning_rate=learning_rate*(1 - 0.2/epoch_tmp)
			for minibatch_index in range(n_train_batches):
				x_value = x_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				y_value = y_train[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				value = train_f(learning_rate, x_value, y_value)
				if np.isnan(value):
					model.initialize()
					reinit()
					best_valid = np.inf
					increment = init_increment
					#import pdb
					#pdb.set_trace()
			valid_cost=[]
			for minibatch_index in range(n_valid):
				x_value = x_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				y_value = y_valid[minibatch_index*batch_size:(minibatch_index+1)*batch_size]
				value = test_f(x_value, y_value)
				valid_cost.append(value)

			# deciding when to stop the training on the sub batch
			valid_result = np.mean(valid_cost)
			if valid_result <= best_valid*0.95:
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
				state_of_train['TRAIN']=best_train
				state_of_train['VALID']=best_valid
				state_of_train['TEST']=best_test;

			else:
				increment-=1

			if not done and increment ==0:
				increment = init_increment
				done = True
				#learning_rate/=2
				#train_f, valid_f, test_f, model, reinit = build_training(lr=learning_rate*0.1, model=model)
			if done and increment==0:
				# keep the best set of params found during training
				model.load_model()
				increment = init_increment
				done = False
				record_state(n_train_batches, n_train, best_train, best_valid, best_test, record_repo, record_filename)
				# record in a file
				print "RATIO :"+str(1.*n_train_batches/n_train*100)
				import time
				start_time = time.clock()
				(x_train, y_train), n_train_batches = active_selection(model, x_train, y_train, n_train_batches, batch_size, valid_f, (x_valid, y_valid), option, nb_to_query, valid_full=state_of_train['VALID'])
				end_time = time.clock()
				print end_time - start_time
				return
				#model.initialize()
				model.initial_state()
				reinit()
				best_valid=np.inf; best_train=np.inf; best_test=np.inf
				learning_rate = lr_init
				epoch_tmp=0
				#train_f, valid_f, test_f, model, reinit = build_training(lr=learning_rate*0.1, model=model)
				#state_of_train['TRAIN']=best_train; state_of_train['VALID']=best_valid; state_of_train['TEST']=best_test; 

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
	# read options
        import resource
        #resource.setrlimit(resource.RLIMIT_CPU, (1,2))
        #resource.setrlimit(resource.RLIMIT_NPROC, (30, 30))
	import sys
        nb_to_query=1
	if len(sys.argv)>=4:
                        option=sys.argv[1];
                        nb_to_query=(int)(sys.argv[2]);
			record_repo=sys.argv[3];
			record_filename=sys.argv[4]
	repo="/home/ducoffe/Documents/Code/datasets/mnist"

	filenames="mnist.pkl.gz"
	learning_rate = 1e-3
	batch_size = 64
	training(repo, learning_rate, batch_size,filenames, option, record_repo, record_filename, nb_to_query)
