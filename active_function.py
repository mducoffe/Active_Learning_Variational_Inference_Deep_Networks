##### FUNCTION RELATED TO ACTIVE LEARNING #####
# @date 24/02/2016 ############################
# @author Melanie Ducoffe #####################
###############################################
from training_mnist import build_training, training_committee
import numpy as np
import theano
import theano.tensor as T

#from kronecker_covering.optimality import opt_T
from kronecker_covering.covering_new import covering_KL


def covering_criterion(model, x_train, y_train, batch_size, n_train_batches, index_to_add_init, nb_to_query):
        print 'MLE'
	n = n_train_batches*batch_size
	data_labelled = (x_train[:n_train_batches*batch_size], y_train[:n_train_batches*batch_size])
	data_unlabelled = x_train[n_train_batches*batch_size:n_train_batches*batch_size + 10*batch_size]
	# build_subset
	shape_x = x_train[0].shape
	subsets = np.zeros((len(index_to_add_init),)+shape_x)
	filters_info = model.L_filter_size
	for i, index in zip(range(len(index_to_add_init)), index_to_add_init):
		subsets[i] = x_train[index]
	indices = covering_KL(subsets, data_labelled, data_unlabelled, model, filters_info, nb_to_query*batch_size)

	# pb subset can me empty : why ???
	#indices = opt_T(subsets, data_labelled, data_unlabelled, model, filters_info, nb_to_query*batch_size)
	return [index_to_add_init[j] for j in indices]
	#return (np.random.permutation(len(y_train) - n) +n)[:nb_to_query*batch_size]


def compute_margin(majority, probabilities):
	# majority is the class index

	if np.argmax(probabilities) == majority:
		return 0.
	else:
		return 1.

	return np.max(probabilities) - probabilities[majority]

def active_selection(model, x_train, y_train, n_train_batches, batch_size, valid_function, valid, option="qbc", nb_to_query=1, query_size=20, nb_classe=10, valid_full=1.):
        
        if query_size - nb_to_query < 10:
             query_size = nb_to_query + 30
	if option =='qbc':
		(x_train, y_train), n_train_batches=active_selection_qbc(model, x_train, y_train, n_train_batches,
									batch_size, valid_function, query_size, nb_classe, valid, valid_full)
	elif option =='random':
		(x_train, y_train), n_train_batches=active_selection_random(x_train, y_train, n_train_batches, batch_size, valid_function, query_size, nb_classe)
	elif option=='uncertainty':
		(x_train, y_train), n_train_batches=active_selection_uncertainty(model, x_train, y_train, n_train_batches, batch_size, nb_to_query)

	elif option=='curriculum':
		(x_train, y_train), n_train_batches=active_selection_curriculum(model, x_train, y_train, n_train_batches, batch_size, nb_to_query)

	elif option=='mle':
		(x_train, y_train), n_train_batches=active_selection_mle(model, x_train, y_train, n_train_batches, batch_size, nb_to_query)
	else:
		raise NotImplementedError()

	return (x_train, y_train), n_train_batches

def active_selection_qbc(model, x_train, y_train, n_train_batches, batch_size, valid_function, query_size=20, nb_classe=10, valid=None, valid_full=1.):

	batch_size = batch_size
	#build the committee
	x = T.tensor4('x')
	committee = model.generate_comitee()
	n_train = len(y_train)/batch_size
	
	#STEP 2
	sub_train = (x_train[:n_train_batches*batch_size], y_train[:n_train_batches*batch_size])
	learning_rate=1e-3
	# JUST FOR TEST
	committee=training_committee(committee, learning_rate, sub_train , batch_size*2, valid, valid_full)
	temp_decision = [ instance.predict(x) for instance in committee] # LABELS
	confidence = [instance.probabilities(x) for instance in committee] # PROBABILITIES
	function = theano.function(inputs=[x], outputs= temp_decision,
					allow_input_downcast=True)
	function_prob = theano.function(inputs=[x], outputs= confidence,
					allow_input_downcast=True)
	minibatches_comitee = (np.random.permutation(n_train - n_train_batches) + n_train_batches)[:query_size]
	#return (x_train, y_train), n_train_batches # FIND THE MISTAKE
	#evaluation
	print 'EVALUATION TO DO !!!'
        minibatches_errors = []
        temp = np.zeros((nb_classe, batch_size*len(minibatches_comitee)), dtype=np.float32)
	temp_prob = np.zeros((len(committee), batch_size*len(minibatches_comitee), nb_classe), dtype=np.float32)

        for j, minibatch in zip(xrange(len(minibatches_comitee)), minibatches_comitee):
        	x_batch = x_train[minibatch*batch_size:(minibatch+1)*batch_size]
		if len(x_batch)==0:
			import pdb
			pdb.set_trace()
			print 'holly shit'
                value = np.asarray(function(x_batch)) # prediction of the network (committee_size, batch_size)
		value_prob = np.asarray(function_prob(x_batch)) #(committee_size, batch_size, nb_class)
		assert value.shape == (len(committee), batch_size), "dimension mismatched for value"
		assert value_prob.shape == (len(committee), batch_size, nb_classe), "dimension mismatched for value_prob"
		for i in xrange(len(committee)):
                	for index in xrange(batch_size):
                        	temp[ value[i,index] , batch_size*j + index]+= 1; # store the occurences
			temp_prob[i, batch_size*j:(j+1)*batch_size, :] = value_prob[i] # store the prediction probs for each instance

	# find the most predicted label
	majority = np.argmax(temp, axis=0)
	if len(majority) != batch_size*len(minibatches_comitee):
		raise Exception('size mismatched in the active learning step')
	for q in xrange(batch_size*len(minibatches_comitee)):
		score = sum([compute_margin(majority[q], temp_prob[i,q,:]) for i in range(len(committee))])
		minibatches_errors.append(score)

	index_to_add_ = np.argsort(minibatches_errors)[::-1][:2*batch_size] # TO TEST MUCH FASTER
	#################### COVERING CRITERION ###########################
	#index_to_add_ = covering_criterion(model, x_train, y_train, batch_size, n_train_batches, index_to_add_)
	###################################################################
	index_to_add = []

	for q in xrange(len(index_to_add_)):
		first_coord = index_to_add_[q]/batch_size # indice du minibatch qu'on va selectionner
		second_coord = index_to_add_[q] - batch_size*first_coord # indice du sample dans le minibatch
		index_to_add.append([minibatches_comitee[first_coord], second_coord])

	for j,coord in zip(xrange(len(index_to_add)), index_to_add):
		coord_0 = coord[0]
		coord_1 = coord[1]
		temp_x = x_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		temp_y = y_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		x_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		y_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_x
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_y
	n_train_batches+=len(index_to_add_)/batch_size;

	return (x_train, y_train), n_train_batches


def active_selection_random(x_train, y_train, n_train_batches, batch_size, valid_function, query_size=20, nb_classe=10):

	batch_size = batch_size
	n_train = len(y_train)/batch_size
	#STEP 2
	minibatches_comitee = (np.random.permutation(n_train - n_train_batches) + n_train_batches)[:query_size]

	#evaluation
	minibatches_errors = np.random.permutation(batch_size*len(minibatches_comitee))
	index_to_add_ = minibatches_errors[:10*batch_size]
	#################### COVERING CRITERION ###########################
	#index_to_add_ = covering_criterion(model, x_train, y_train, batch_size, n_train_batches, index_to_add_)
	###################################################################
	index_to_add = []

	for q in xrange(10*batch_size):
		first_coord = index_to_add_[q]/batch_size # indice du minibatch qu'on va selectionner
		second_coord = index_to_add_[q] - batch_size*first_coord # indice du sample dans le minibatch
		index_to_add.append([minibatches_comitee[first_coord], second_coord])

	for j,coord, score in zip(xrange(len(index_to_add)), index_to_add, np.sort(minibatches_errors)[::-1][:batch_size]):
		coord_0 = coord[0]
		coord_1 = coord[1]
		temp_x = x_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		temp_y = y_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		x_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		y_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_x
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_y
	n_train_batches+=1;
	return (x_train, y_train), n_train_batches

def active_selection_uncertainty(model, x_train, y_train, n_train_batches, batch_size, nb_to_query=1, query_size=20):
	print('UNCERTAINTY')
	#build the committee
	x = T.tensor4('x')
	n_train = len(y_train)/batch_size
	confidence = theano.function([x], model.confidence(x), allow_input_downcast=True) # TO CHECK
	minibatches_comitee = (np.random.permutation(n_train - n_train_batches) + n_train_batches)[:query_size]
	shape = x_train[0].shape #(nb channel, width, height)
	uncertainty = []
	for index in minibatches_comitee:
		uncertainty.append(confidence(x_train[index*batch_size:(index+1)*batch_size])) # max prediction
	batches_confidence = np.concatenate(uncertainty, axis=0)
	#index_to_add_ = np.argsort(batches_confidence)[nb_to_query*batch_size:] # confiance la plus faible # curriculum
	index_to_add_ = np.argsort(batches_confidence)[-nb_to_query*batch_size::] # confiance la plus faible (uncertainty)
	#################### COVERING CRITERION ###########################
	#index_to_add_ = covering_criterion(model, x_train, y_train, batch_size, n_train_batches, index_to_add_, nb_to_query) # TEST
	###################################################################
	index_to_add = []
	for q in xrange(nb_to_query*batch_size):
		first_coord = index_to_add_[q]/batch_size # indice du minibatch qu'on va selectionner
		second_coord = index_to_add_[q] - batch_size*first_coord # indice du sample dans le minibatch
		if first_coord <0 or first_coord >= len(minibatches_comitee):
			print 'PROBLEM'
			print first_coord
		index_to_add.append([minibatches_comitee[first_coord], second_coord])

	for j,coord in zip(xrange(len(index_to_add)), index_to_add):
		coord_0 = coord[0]
		coord_1 = coord[1]
		temp_x = x_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		temp_y = y_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		x_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		y_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_x
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_y
	n_train_batches+=nb_to_query;
	return (x_train, y_train), n_train_batches

def active_selection_curriculum(model, x_train, y_train, n_train_batches, batch_size, nb_to_query=1, query_size=20):

	batch_size = batch_size
	#build the committee
	x = T.tensor4('x')
	n_train = len(y_train)/batch_size
	confidence = theano.function([x], model.confidence(x), allow_input_downcast=True) # TO CHECK
	minibatches_comitee = (np.random.permutation(n_train - n_train_batches) + n_train_batches)[:query_size]
	shape = x_train[0].shape #(nb channel, width, height)
	uncertainty = []
	for index in minibatches_comitee:
		uncertainty.append(confidence(x_train[index*batch_size:(index+1)*batch_size])) # max prediction

	batches_confidence = np.concatenate(uncertainty, axis=0)
	index_to_add_ = np.argsort(batches_confidence)[:nb_to_query*batch_size] # confiance la plus forte (curriculum)
	#################### COVERING CRITERION ###########################
	#index_to_add_ = covering_criterion(model, x_train, y_train, batch_size, n_train_batches, index_to_add_, nb_to_query) # TEST
	###################################################################
	index_to_add = []
	for q in xrange(nb_to_query*batch_size):
		first_coord = index_to_add_[q]/batch_size # indice du minibatch qu'on va selectionner
		second_coord = index_to_add_[q] - batch_size*first_coord # indice du sample dans le minibatch
		if first_coord <0 or first_coord >= len(minibatches_comitee):
			print 'PROBLEM'
			print first_coord
		index_to_add.append([minibatches_comitee[first_coord], second_coord])

	for j,coord in zip(xrange(len(index_to_add)), index_to_add):
		coord_0 = coord[0]
		coord_1 = coord[1]
		temp_x = x_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		temp_y = y_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		x_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		y_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_x
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_y
	n_train_batches+=nb_to_query;
	return (x_train, y_train), n_train_batches

def active_selection_mle(model, x_train, y_train, n_train_batches, batch_size, nb_to_query=1, query_size=3):

	#################### COVERING CRITERION ###########################
	n = n_train_batches*batch_size
	index_to_add_init = (np.random.permutation(len(y_train) - n) +n)[:query_size*batch_size]
	index_to_add_ = covering_criterion(model, x_train, y_train, batch_size, n_train_batches, index_to_add_init, nb_to_query)
	###################################################################
	index_to_add = []
	for q in xrange(nb_to_query*batch_size):
		first_coord = index_to_add_[q]/batch_size # indice du minibatch qu'on va selectionner
		second_coord = index_to_add_[q] - batch_size*first_coord # indice du sample dans le minibatch
		if first_coord <0 or first_coord > len(y_train)/batch_size:
			print 'PROBLEM'
			print first_coord
			import pdb
			pdb.set_trace()
		index_to_add.append([first_coord, second_coord])

	for j,coord in zip(xrange(len(index_to_add)), index_to_add):
		coord_0 = coord[0]
		coord_1 = coord[1]
		temp_x = x_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		temp_y = y_train[n_train_batches*batch_size + j:(n_train_batches)*batch_size + j+1]
		x_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		y_train[n_train_batches*batch_size + j:n_train_batches*batch_size + j+1]=\
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1]
		x_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_x
		y_train[coord_0*batch_size + coord_1:coord_0*batch_size + coord_1 + 1] = temp_y
	n_train_batches+=nb_to_query;
	return (x_train, y_train), n_train_batches
