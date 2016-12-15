#### per blocks Kronecker approximations (KFAC) ######
# @author Melanie Ducoffe ############################
# @date 26/04/2016 ###################################
######################################################
import numpy as np
#from invert_matrix import approximate_inverse
from nys import approximate_inverse


"""
def minibatch_product(M, batch_size, mean=True):
	# M => (minibatch, matrix)
	if M.ndim == 2:
		M = M.reshape((M.shape[0], 1, M.shape[1]))

	if mean:
		M = np.mean(M, axis=0)
		M_T = np.transpose(M)
		return np.dot(M_T, M)
	else:
		M_T = M.transpose((0,2,1))

		R = None
		for i in range(batch_size):
			if R is None:
				R = np.dot(M_T[i], M[i])
			else:
				R += np.dot(M_T[i], M[i])
		return R
"""

def kfac_labelled(X, Y, f, g, batch_size=32, dico=None):
	batch_size = 32
	dico = {}
	if len(X) < batch_size:
		batch_size = len(X)
	for minibatch in range(len(Y)/batch_size):
		# TO DO : need for reshape ???
		x_batch = X[minibatch*batch_size:(minibatch+1)*batch_size]
		y_batch = Y[minibatch*batch_size:(minibatch+1)*batch_size]
		dico_batch = f(x_batch, y_batch) # theano function + preprocessing blocks
		for key in dico_batch :
			"""
			if key[:-2]=="conv_output":
				dico_batch[key] = np.transpose(dico_batch[key], (1,0, 2,3))
				shape = dico_batch[key].shape
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1], shape[2]*shape[3]))
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1]*shape[2]*shape[3]))
				dico_batch[key] = np.transpose(dico_batch[key])
			"""
			if not(key) in dico:
				dico[key] = g(dico_batch[key])
			else:
				dico[key] += g(dico_batch[key])
			
		del dico_batch

	# if len(Y) is not a multiple of batch_size, still take in account the last samples
	if len(Y) % batch_size !=0:
		minibatch = len(Y)/batch_size
		x_batch = X[minibatch*batch_size:]
		y_batch = Y[minibatch*batch_size:]
		dico_batch = f(x_batch, y_batch) # theano function + preprocessing blocks
		for key in dico_batch :
			"""
			if key[:-2]=="conv_output":
				dico_batch[key] = np.transpose(dico_batch[key], (1,0, 2,3))
				shape = dico_batch[key].shape
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1], shape[2]*shape[3]))
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1]*shape[2]*shape[3]))
				dico_batch[key] = np.transpose(dico_batch[key])
			"""
			if not(key) in dico:
				dico[key] = g(dico_batch[key])
			else:
				dico[key] += g(dico_batch[key])

	return dico

def kfac_query(X, f_u, batch_size, dico=None):

	dico_batch = f(x_batch)
	queries = []
	for m in range(len(X)):
		empty_dico = dico([ (key, np.zeros_like(dico_batch[key][0])) for key in dico_batch.keys()])
		queries.append(empty_dico)
	return
	"""
	for key in dico_batch.keys():
		dico_batch[key] = minibatch_product(dico_batch[key], batch_size, False)
		
	for m in range(len(X)):
	"""
		
def kfac_unlabelled(X, f, g, batch_size=1024, dico=None):
	batch_size = 32
	dico = {}
	if len(X) < batch_size:
		batch_size = len(X)
	for minibatch in range(len(X)/batch_size):
		# TO DO : need for reshape ???
		x_batch = X[minibatch*batch_size:(minibatch+1)*batch_size]
		dico_batch = f(x_batch) # theano function + preprocessing blocks
		for key in dico_batch :
			"""
			if key[:-2]=="conv_output":
				dico_batch[key] = np.transpose(dico_batch[key], (1,0, 2,3))
				shape = dico_batch[key].shape
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1], shape[2]*shape[3]))
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1]*shape[2]*shape[3]))
				dico_batch[key] = np.transpose(dico_batch[key])
			"""
			if not(key) in dico:
				dico[key] = g(dico_batch[key])
			else:
				dico[key] += g(dico_batch[key])
			
		del dico_batch

	# if len(Y) is not a multiple of batch_size, still take in account the last samples
	if len(X) % batch_size !=0:
		minibatch = len(X)/batch_size
		x_batch = X[minibatch*batch_size:]
		dico_batch = f(x_batch) # theano function + preprocessing blocks
		for key in dico_batch :
			"""
			if key[:-2]=="conv_output":
				dico_batch[key] = np.transpose(dico_batch[key], (1,0, 2,3))
				shape = dico_batch[key].shape
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1], shape[2]*shape[3]))
				dico_batch[key]=dico_batch[key].reshape((shape[0], shape[1]*shape[2]*shape[3]))
				dico_batch[key] = np.transpose(dico_batch[key])
			"""
			if not(key) in dico:
				dico[key] = g(dico_batch[key])
			else:
				dico[key] += g(dico_batch[key])

	return dico


def blocks_Fisher_Q(X, f_u, g):
	dico = f_u(X)
	for key in dico.keys():
		dico[key] = g(dico[key])
	import pdb
	pdb.set_trace()


# blocks for the Fisher of all data
def blocks_Fisher_P(X_L, Y_L, X_U, f_l, f_u, g, batch_size=512*2):
	#print 'dico_l'
	dico_l = kfac_labelled(X_L, Y_L, f_l, g, batch_size)
	#print 'dico_u'
	dico_u = kfac_unlabelled(X_U, f_u, g, batch_size)
	dico = {}
	#print 'union dico_l and dico_u'
	mean_inv = 1./(len(X_L)+len(X_U))
	for key in dico_l:
		if key!="logistic_input" and key!="logistic_output":
			dico[key] = mean_inv*(dico_l[key]+ dico_u[key])
		else:
			# special preprocessing for logistic layer !
			dico[key] = dico_l[key]+dico_u[key]
			raise NotImplementedError("logistic is not done yet but soon !")

	"""
	if 'logistic_input' in dico.keys():
		dico['logistic_input'], dico['logistic_output'] = kronecker_logistic_decomposition(dico['logistic_input'],
											   	dico['logistic_output'])
	"""
	#print 'inverse'
	for key in dico:
		shape = dico[key].shape[0]
		dico[key] = approximate_inverse(dico[key], np.min([shape, 2000]))

	return dico, dico_l


"""
def blocks_Fisher_Q(X, f_u):
	# for one sample only : for memory consumption we compute the coefficient directly
	assert len(X)==1, "one sample only is required for memory consumption"
	query = kfac_unlabelled(X, f_u, 1)
	if "logistic_input" in query:
		query["logistic_input"] = query["logistic_input"][0][0]
		query["logistic_output"] = query["logistic_output"][0][0]
	return query
"""

