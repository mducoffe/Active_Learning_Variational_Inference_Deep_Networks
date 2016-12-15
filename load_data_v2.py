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
import h5py
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


def build_dataset(repo, filename="svhn_format_2.hdf5"):

	with closing(h5py.File(os.path.join(repo, filename), mode='r')) as f:
		print f.items()
		x = np.array(f['features'])
		y = np.array(f['targets'])
	
	print 'kikou'
	x,y = preprocessing(x,y)
	# save
	g = h5py.File(os.path.join(repo, "svhn_zca.pkl"), mode='w')
	g.create_dataset("features", data=x)
	g.create_dataset("targets", data=y)


def preprocessing(x,y):
	f = lcn_function(radius=7)

	z = np.zeros_like(x, dtype=np.float32)
	z[:,0,:,:] = 0.299*x[:,0,:,:] + 0.587*x[:,1,:,:] + 0.114*x[:,2,:,:]
	z[:,1,:,:] = -0.14713*x[:,0,:,:] - 0.28886*x[:,1,:,:] + 0.436*x[:,2,:,:]
	z[:,2,:,:] = 0.615*x[:,0,:,:] - 0.51498*x[:,1,:,:] - 0.10001*x[:,2,:,:]
	# 14:21
	batch_size = 100
	for i in range(len(y)/batch_size):
		print i, len(y)/batch_size
		z[i*batch_size:(i+1)*batch_size,0:1,:,:] = f(z[i*batch_size:(i+1)*batch_size,0:1,:,:])
	
	return z,y


if __name__=="__main__":
	repo="/home/ducoffe/Documents/Code/datasets/svhn"
	build_dataset(repo)
