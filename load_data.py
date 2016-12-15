###### LOADING SVHN ############
# @author Melanie Ducoffe
# @date 22/02/2016
################################
import scipy.io
from contextlib import closing
import os
import numpy as np
from lcn import lcn_function
import pickle
import gzip

def shuffle(x, y):
	index = np.random.permutation(len(y))
	x_ = np .zeros_like(x)
	y_ = np.zeros_like(y)

	for i in range(len(y)):
		x_[i] = x[index[i]]
		y_[i] = y[index[i]]

	return x_, y_

def split(x, y, percentage):
	n = len(y)
	n_train = n - (int) (n*percentage)
	return (x[:n_train], y[:n_train]), (x[n_train:], y[n_train:])

def load_dataset_usps(repo, filename):

	mat = scipy.io.loadmat(os.path.join(repo, filename))
	x_train = mat['train_patterns']
	x_test = mat['test_patterns']
	y_train = mat['train_labels']
	y_test = mat['test_labels']

	# processing
	x_train = x_train.transpose((1,0))
	x_shape = x_train.shape
	x_train = x_train.reshape((x_shape[0], 1, 16, 16))

	x_test = x_test.transpose((1,0))
	x_shape = x_test.shape
	x_test = x_test.reshape((x_shape[0], 1, 16, 16))

	y_train = np.argmax(y_train, axis=0)[:,None]
	y_test = np.argmax(y_test, axis=0)[:,None]

	## create validation with 10% of the train
	# step 1 : shuffle
	x_train, y_train = shuffle(x_train, y_train)

	# step2 : split
	(x_train, y_train), (x_valid, y_valid) = split(x_train, y_train, 0.1)

	return (x_train, y_train), (x_valid, y_valid), (x_test, y_test)

"""

-- For natural images, we use several intuitive tricks:
--   + images are mapped into YUV space, to separate luminance information
--     from color information
--   + the luminance channel (Y) is locally normalized, using a contrastive
--     normalization operator: for each neighborhood, defined by a Gaussian
--     kernel, the mean is suppressed, and the standard deviation is normalized
--     to one.
--   + color channels are normalized globally, across the entire dataset;
--     as a result, each color component has 0-mean and 1-norm across the dataset.
"""

def private_normalize(X):
	"""
	# save data in a temporary file, check does not exist
	tmp_filename="tmp_repo/temp_filename_"
	tmp_index=0
	# divide in 4 parts
	while os.path.isfile(tmp_filename+str(tmp_index)):
		tmp_index+=1
	tmp_filename+=str(tmp_index)
	value=0
	increment= 1000
	length = len(X)/increment
	if length*increment != len(X):
		length+=1
	for i in range(length):
		with closing(open(tmp_filename+"_v"+str(i), 'wb')) as f:
			data = X[value:value+increment,:,:,:]
			pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
			value+=increment
		del f
	"""
	# work on data as if it was std
	mean = np.mean(X, axis=0); mean=mean.reshape((1,3,32,32))
	return mean, mean
	
	# compute std 'by hand' to avoid memory allocation error
	X -= mean
	return mean, mean
	X**=2
	X = np.mean(X, axis=0);
	X = np.sqrt(X);
	std=X.reshape((1,3,32,32))
	return mean, std
	"""
	del X
	print 'OKAY !!!!'
	# reload X and delete temporary files
	x_train=[]
	for i in range(length):
		print i
		if i>=390:
			import pdb
			pdb.set_trace()
		with closing(open(tmp_filename+"_v"+str(i), 'rb')) as f:
			x_value = pickle.load(f)
			x_train.append(x_value)
			del x_value
		del f
		#os.remove(tmp_filename+"_v"+str(i))
	print 'hhhhh'
	X = np.concatenate(x_train, axis=0)
	del x_train
	return X, mean, std
	"""

def build_dataset(repo, filenames, mean=None, std=None):
	f = lcn_function(radius=7)
	if type(filenames)!=list:
		raise Exception('unknow type for filename')
	print 'LOADING DATA'
	x_trains=[]; y_trains=[]
	for filename in filenames:
		mat = scipy.io.loadmat(os.path.join(repo, filename))
		x_train = mat['X'].transpose()
		x_train = x_train.astype(np.float32)
		z_train = np.zeros_like(x_train)
		print 'YUV space'
		# mapping into YUV space to separate luminance information from the color information
		z_train[:,0,:,:] = 0.299*x_train[:,0,:,:] + 0.587*x_train[:,1,:,:] + 0.114*x_train[:,2,:,:]
		z_train[:,1,:,:] = -0.14713*x_train[:,0,:,:] - 0.28886*x_train[:,1,:,:] + 0.436*x_train[:,2,:,:]
		z_train[:,2,:,:] = 0.615*x_train[:,0,:,:] - 0.51498*x_train[:,1,:,:] - 0.10001*x_train[:,2,:,:]
		x_trains.append(z_train); y_trains.append(mat['y'])
		del z_train; del x_train; del mat;

	Y = np.concatenate(y_trains, axis=0)
	print 'ok for Y'
	ensemble_dim = sum([len(x_trains[i]) for i in range(len(x_trains))])
	print str(ensemble_dim)+" training samples"
	X = np.zeros((ensemble_dim, 3, 32, 32))
	index=0
	for x_train in x_trains:
		shape = len(x_train)
		X[index:index+shape] = x_train
		index+=shape
	print 'ok for X'
	print 'calling the garbage collector'
	del x_trains; del y_trains
	print 'LCN'
	# contrastive normalization across the luminance channel
	#f = lcn_function(radius=7)
	for i in range(len(X)):
		X[i,0,:,:] = f(X[i, 0, :,:].reshape((1,1,32,32)))
	X = X.astype(np.float32)
	del f
	# preprocessing data: normalize each feature (channel) globally
	print 'GLOBAL NORMALIZATION'
	if (mean is None and std is None):
		#mean, std = private_normalize(X)
		mean = np.mean(X, axis=0).reshape((1,3,32,32))
		std = np.std(X, axis=0).reshape((1, 3, 32, 32)) 
	X -= mean; X/=std;
	print 'SUCCEED'
	print 'check mean'
	return (X, Y), (mean, std)

def build_datasets(repo, filename_train=["train_32x32.mat", "extra_32x32.mat"], filename_test=["test_32x32.mat"], filename_to_save="svhn_pickle_v3"):
	print "TRAINING"
	(x_train, y_train), (mean,std) = build_dataset(repo, filename_train)
	"""
	with closing(open(os.path.join(repo, filename_to_save+"_train"), 'wb')) as f:
		pickle.dump((x_train, y_train), f, protocol = pickle.HIGHEST_PROTOCOL)
	"""
	import h5py
	f_x_train = h5py.File(os.path.join(repo, filename_to_save+"_x_train"), mode='w')
	f_x_train.create_dataset("init", data=x_train)
	f_y_train = h5py.File(os.path.join(repo, filename_to_save+"_y_train"), mode='w')
	f_y_train.create_dataset("init", data=y_train)
	f_x_train.close(); f_y_train.close()
	del x_train; del y_train
	print "TESTING "
	(x_test, y_test), _ = build_dataset(repo, filename_test, mean, std)
	f_x_test = h5py.File(os.path.join(repo, filename_to_save+"_x_test"), mode='w')
	f_x_test.create_dataset("init", data=x_test)
	f_y_test = h5py.File(os.path.join(repo, filename_to_save+"_y_test"), mode='w')
	f_y_test.create_dataset("init", data=y_test)
	f_x_test.close(); f_y_test.close()
	"""
	with closing(open(os.path.join(repo, filename_to_save+"_test"), 'wb')) as f:
		pickle.dump((x_test, y_test), f, protocol = pickle.HIGHEST_PROTOCOL)
	"""

def load_datasets_mnist(repo, filename):
	with closing(gzip.open(os.path.join(repo, filename), 'rb')) as f:
		train_set, valid_set, test_set = pickle.load(f)
	x_train, y_train = train_set
	x_valid, y_valid = valid_set
	x_test, y_test = test_set
	x_train = x_train.reshape((x_train.shape[0], 1, 28, 28))
	x_valid = x_valid.reshape((x_valid.shape[0], 1, 28, 28))
	x_test = x_test.reshape((x_test.shape[0], 1, 28, 28))
	y_train = y_train.reshape((y_train.shape[0], 1))
	y_valid = y_valid.reshape((y_valid.shape[0], 1))
	y_test = y_test.reshape((y_test.shape[0], 1))
	train_set = (x_train, y_train)
	valid_set = (x_valid, y_valid)
	test_set = (x_test, y_test)
	return train_set, valid_set, test_set


def load_datasets(repo, filenames="mnist.pkl.gz"):
	import h5py
	# first chek keys in the dictionnary
	assert 'x_train' in filenames.keys()
	assert 'y_train' in filenames.keys()
	assert 'x_test' in filenames.keys()
	assert 'y_test' in filenames.keys()
	for key in filenames.keys():
		fname = os.path.join(repo, filenames['x_train'])
		if not os.path.isfile(fname) :
			raise Exception('unknown file : %s', fname)
	with closing(h5py.File(os.path.join(repo, filenames['x_train']), 'r')) as f:
		x_train = np.asarray(f.items()[0][1]).astype(np.float32)
	with closing(h5py.File(os.path.join(repo, filenames['y_train']), 'r')) as f:
		y_train = np.asarray(f.items()[0][1])-1
	with closing(h5py.File(os.path.join(repo, filenames['x_test']), 'r')) as f:
		x_test = np.asarray(f.items()[0][1]).astype(np.float32)
	with closing(h5py.File(os.path.join(repo, filenames['y_test']), 'r')) as f:
		y_test = np.asarray(f.items()[0][1])-1
	
	# shuffle elements
	nb_train = len(y_train)
	index = np.random.permutation(nb_train)
	for i, j in zip(range(nb_train), index):
		tmp_x = x_train[i]; tmp_y = y_train[i]
		x_train[i]=x_train[j]
		x_train[j]=tmp_x
		y_train[i]=y_train[j]
		y_train[j]=tmp_y
	percentage = (int) (0.9*nb_train)
	return ((x_train[:percentage], y_train[:percentage]), 
		(x_train[percentage:], y_train[percentage:]),
		(x_test, y_test))
	

def pickle_datasets(repo, filename="svhn_pickle_v2"):
	with closing(open(os.path.join(repo, filename), 'rb')) as f:
		train, test = pickle.load(f)
	x_train, y_train = train; y_train-=1
	(x_test, y_test) = test; y_test-=1

	# took 10% of training data to build validation
	index = np.random.permutation(len(y_train))

 	# shuffle elements
	for j,i in zip(range(len(y_train)), index):
		tmp_x = x_train[j]
		x_train[j] = x_train[i]
		x_train[i] = tmp_x
		tmp_y = y_train[j]
		y_train[j] = y_train[i]
		y_train[i] = tmp_y
	percentage = (int) (0.95*len(y_train))
	x_valid, y_valid = x_train[percentage:], y_train[percentage:]
	x_train, y_train = x_train[:percentage], y_train[:percentage]
	return (x_train, y_train), (x_train, y_train), (x_test, y_test)

if __name__=="__main__":
	repo="/home/ducoffe/Documents/Code/datasets/svhn"
	build_datasets(repo)

	"""
	filenames={}; 
	filenames["x_train"]="svhn_pickle_v3_x_train";
	filenames["y_train"]="svhn_pickle_v3_y_train";
	filenames["y_test"]="svhn_pickle_v3_y_test";
	filenames["x_test"]="svhn_pickle_v3_x_test";
	train, valid, test = load_datasets(repo, filenames)
	import pdb
	pdb.set_trace()
	"""
