#### test sampling with normal distribution
import numpy as np


def generate_sample(N):
	return np.array([np.random.ranf(1)[0] for n in range(N)])


def build_sample_layer(cov_input, cov_output, multiply=None):
	A = cov_input.shape[0]; B = cov_output.shape[0]
	x_s = [ generate_sample(B) for n in range(A)]
	X = np.array(x_s) # [X_1, X_2, ..., X_n] #shape(A,B)
	X = X.transpose((1,0))

	# (B kro A)vec X=C equiv AXB.T = C
	Y = np.dot(np.dot(cov_output, X), cov_input.transpose((1,0)))
	# vec
	Y = Y.reshape((-1,1), order="F")
	Z = X.reshape((-1,1), order="F")

	
	C = np.kron(cov_input,cov_output)
	print C.shape, Z.shape
	D = np.dot(C, Z)

	print np.min(D -Y)
	print np.max(D-Y)

if __name__=="__main__":
	cov_input=np.ones((9,9))
	cov_output=np.random.ranf((20, 20))
	build_sample_layer(cov_input, cov_output)
