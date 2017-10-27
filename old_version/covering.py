######### covering criterions ######
# @date 26/04/2016            ######
# @author Melanie Ducoffe ##########
####################################
import numpy as np
from retrieve_params import (preprocessing_blocks,
			     build_intermediate_var_labelled,
			     build_intermediate_var_unlabelled,
			     mul_theano)

from kronecker_factor import blocks_Fisher_P, blocks_Fisher_Q

# sum up function to apply on labelled data
def build_f_l(model, filters_info):
	func = None
	f = build_intermediate_var_labelled(model)

	def func(X,Y):
		dico = f(X,Y)
		return preprocessing_blocks(dico, filters_info)

	return func


# sum up function to apply on unlabelled data
def build_f_u(model, filters_info):
	func = None
	f = build_intermediate_var_unlabelled(model)

	def func(X):
		dico = f(X)
		return preprocessing_blocks(dico, filters_info)

	return func


# greedy selection
# compute Tr(X_n), Tr(Y_n), Tr(X_n*Y_n), Tr(X_n*C^-1), Tr(Y_n*D^-1), Tr(X_n*C^-1)Tr(Y_n*D^-1)
# for every blocks and add them for one single list per sample
# there is a C for every block !

def get_initial_coeffs(dico_l, dico_1):
	coeffs_qeries = []
	keys = dico_1.keys()
	# create couples for input/output layers :
	couple_key=[]
	if "logistic_input" in keys:
		couple_key=[("logistic_input", "logistic_output")]
	for index in range(len(keys)/2):
		if "conv_input_"+str(index) in keys:
			couple_key.append(("conv_input_"+str(index), "conv_output_"+str(index)))
		if "fully_input_"+str(index) in keys:
			couple_key.append(("fully_input_"+str(index), "fully_output_"+str(index)))
	return get_coeff_sample(dico_l, dico_1, couple_key)


def get_coeff_subsets(X, dico_1, f_u, g, batch_size=64):
	coeff_queries = []
	keys = dico_1.keys()
	# create couples for input/output layers :
	couple_key=[]
	if "logistic_input" in keys:
		couple_key=[("logistic_input", "logistic_output")]
	for index in range(len(keys)/2):
		if "conv_input_"+str(index) in keys:
			couple_key.append(("conv_input_"+str(index), "conv_output_"+str(index)))
		if "fully_input_"+str(index) in keys:
			couple_key.append(("fully_input_"+str(index), "fully_output_"+str(index)))
	assert len(couple_key)*2 == len(keys), "bad size for key arguments"

	n = len(X)

	# here do one sample at a time
	for index in range(n):
		x_sample = X[index:index+1]
		dico= f_u(x_sample)
		for key in dico.keys():
			dico[key] = g(dico[key])
			if dico[key].ndim!=2:
				print 'check'
				import pdb
				pdb.set_trace()
		coeff_queries.append((index, get_coeff_sample(dict([(key, dico[key]) for key in dico.keys()]), dico_1, couple_key)))
		del dico

	return coeff_queries
	"""
	for index in range(n):
		X_batch = X[index*batch_size:(index+1)*batch_size]
		dico = f_u(X_batch)
		for key in dico.keys():
			dico[key] = g(dico[key])
		for p in range(batch_size):
			coeff_queries.append((index*batch_size+p, get_coeff_sample(dict([(key, dico[key][p]) for key in dico.keys()]), dico_1, couple_key)))
		del dico
	
	# if one more batch left
	minibatch = len(X) - n*batch_size
	if minibatch !=0:
		X_batch = X[n*batch_size:]
		dico = f_u(X_batch)
		for key in dico.keys():
			dico[key] = g(dico[key])
		for p in range(minibatch):
			coeff_queries.append((n*batch_size+p, get_coeff_sample(dict([(key, dico[key][p]) for key in dico.keys()]), dico_1, couple_key)))
		del dico
	return coeff_queries
	"""


def get_coeff_sample(query, dico_1, couple_keys):
	index =0
	coeffs = [0.]*3
	keys =  couple_keys
	for (key_input, key_output) in keys:
		if query[key_input].ndim==1:
			import pdb
			pdb.set_trace()
		c_n = np.trace( np.dot(query[key_input], dico_1[key_input]))
		d_n = np.trace( np.dot(query[key_output], dico_1[key_output]))
		coeffs[0] +=c_n; coeffs[1] += d_n; coeffs[2]+=c_n*d_n

	return coeffs
	"""
	coeffs = [0.]*6
	keys = couple_keys
	for (key_input, key_output) in keys:
		
		x_n = np.trace(query[key_input])
		y_n = np.trace(query[key_output])
		coeffs[0] += x_n; coeffs[1] += y_n; coeffs[2] += x_n*y_n
		c_n = np.trace( np.dot(query[key_input], dico_1[key_input]))
		d_n = np.trace( np.dot(query[key_output], dico_1[key_output]))
		coeffs[3] += c_n; coeffs[4] += d_n; coeffs[5] += c_n*d_n

	return coeffs
	"""


# query = (int_id, coeffs)
# subsets = ( [ids], sum of coeffs)
def pick_best_query(queries, subsets, N):
	#print 'query', len(queries)
	best_score = -np.inf
	ids, sub_coeffs = subsets
	index = -1
	trace = sub_coeffs[2]
	alpha = 1./(len(subsets)+1) # mean of the subset plus the new sample
	for i in range(len(queries)) :
		id_q, coeffs = queries[i]
		# TO CHECK no abs for the trace !!!
		trace_i = coeffs[2]
		score = N*np.abs( np.log(np.abs(alpha*(trace + trace_i))) -np.log(N)) + alpha*(trace + trace_i)
		"""
		score = alpha*np.log( np.abs(coeffs[2] + coeffs[0]*sub_coeffs[1] + coeffs[1]*sub_coeffs[0] + sub_coeffs[2]))
		score += beta*( coeffs[5] + coeffs[3]*sub_coeffs[4] + coeffs[4]*sub_coeffs[3] + sub_coeffs[5])

		score = alpha*((1+ sub_coeffs[1])*coeffs[0] + (1+sub_coeffs[0])*coeffs[1] 
				+sub_coeffs[2])
		score += beta*((1+ sub_coeffs[4])*coeffs[3] + (1+sub_coeffs[3])*coeffs[4] 
				+sub_coeffs[5])
		"""
		if score > best_score:
			best_score = score; index = i

	# update subset
	if index >= len(queries) or index==-1:
		print 'PROBLEM'
		import pdb
		pdb.set_trace()
		print 'kikou'
	id_q, coeffs = queries[index]
	ids.append(id_q)
	new_sub_coeffs = [0]*3
	new_sub_coeffs[0] = sub_coeffs[0] + coeffs[0]
	new_sub_coeffs[1] = sub_coeffs[1] + coeffs[1]
	new_sub_coeffs[2] = new_sub_coeffs[0]*new_sub_coeffs[1]
	# remove the added sample from the possible queries
	queries = queries[:index] + queries[index+1:]
	return queries, (ids, new_sub_coeffs)
	"""
	new_sub_coeffs = [0]*6
	new_sub_coeffs[0] = sub_coeffs[0] +coeffs[0]
	new_sub_coeffs[1] = sub_coeffs[1] +coeffs[1]
	new_sub_coeffs[3] = sub_coeffs[3] +coeffs[3]
	new_sub_coeffs[4] = sub_coeffs[4] +coeffs[4]
	new_sub_coeffs[2] = sub_coeffs[0]*sub_coeffs[1]
	new_sub_coeffs[5] = sub_coeffs[3]*sub_coeffs[4]
	queries = queries[:index]+queries[index+1:]
	return queries, (ids, new_sub_coeffs)
	"""


def greedy_selection(data, dico_l, dico_1, f_u, size, g, N):
	# size is the size of the subset recquire
	# init subsets score with the score from the labelled training set
	subsets = ([], get_initial_coeffs(dico_l, dico_1))
	queries = get_coeff_subsets(data, dico_1, f_u, g)
	for i in range(size):
		(queries, subsets) = pick_best_query(queries, subsets, N)
	return subsets[0] # list of ids for queries


def covering_KL(subsets, data_labelled, data_unlabelled, model, filters_info, size):

	N = model.nb_parameters()
	f_l = build_f_l(model, filters_info)
	f_u = build_f_u(model, filters_info)


	g = mul_theano()
	# retrieve kronecker blocks for the whole dataset (P distribution)
	X_L, Y_L = data_labelled
	X_U = data_unlabelled
	#print 'block Fisher'
	dico_1, dico_l = blocks_Fisher_P(X_L, Y_L, X_U, f_l, f_u, g)
	#query blocks
	#print 'GREEDY query'
	
	indices = greedy_selection(subsets, dico_l, dico_1, f_u, size, g, N) # TO DO
	return indices
