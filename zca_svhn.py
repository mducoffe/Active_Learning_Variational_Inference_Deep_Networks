##### ZCA normalization for SVHN #######
#@author Melanie Ducoffe ###############
#@date 12/05/2016 ######################
########################################
import pickle
from contextlib import closing

def load_data(repo, pickle_f):
	with closing(open( os.path.join(repo, pickle_f), 'r')) as f:
		(x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pickle.load(f)
	

if __name__=="__main__":
	repo=
	pickle_f=
	load_data(repo, pickle_f)



