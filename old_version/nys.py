######################
#Nystrom method for PSD matrix K
#@author Melanie Ducoffe
#@date 5/04/2016
######################
import numpy as np
#import numpy.linalg as lin
import scipy.linalg as lin


def mul_mat(M):
	assert type(M) is list
	assert len(M)> 1
	A = np.dot(M[0], M[1])
	for B in M[2:]:
		A = np.dot(A, B)
	return A


def mp(S, S_tilde):
	A = np.diag(np.zeros((len(S_tilde),),dtype=S.dtype))
	B = np.zeros((len(S_tilde),), dtype=S.dtype)
	s_t = np.transpose(S)
	for i in range(len(S_tilde)):
		s_i = S_tilde[i]
		B[i] = np.trace(np.dot(s_i, s_t))
		s_i_t = np.transpose(s_i)
		for j in range(i):
			s_j = S_tilde[j]
			A[i,j] = np.trace(np.dot(s_j, s_i_t))
			A[j,i] = A[i,j]

	#tmp = mul_mat([ lin.inv(np.dot(np.transpose(A), A)), np.transpose(A), A])
	Id = np.diag(np.ones((len(S_tilde),), dtype=S.dtype))
	coeffs = mul_mat([ lin.inv(np.dot(np.transpose(A), A)), np.transpose(A), B])
	return coeffs

def inverse_woodburry(A_1, B, c, D):
	#assuming C is diagonal with non zero diagonal coefficient
	import pdb
	pdb.set_trace()
	C_1 = np.diag([1./c_i for c_i in c])
	Z_1 = lin.inv(C_1 + mul_mat([D, A_1, B]))
	Q = mul_mat([A_1, B, Z_1, D, A_1])
	return A_1 - Q
	return A_1 - mul_mat([A_1, B, Z_1, D, A_1])

	"""
	M = generate_random_invert_symm(n)
	(C, v, s_1, u) = nystrom(M, n)
	B = np.dot(C, np.transpose(v))
	D = np.dot(np.transpose(u), np.transpose(C))
	C = np.diag(s_1)

	#R = lin.inv((A + M))
	R = lin.inv(M)
	A_1 = alpha*np.diag(np.ones((n,), dtype=np.float64))
	C_1 = np.diag([1./s_i for s_i in s_1])
	R_tilde = A_1 - mul_mat([A_1, B, lin.inv(C_1 + mul_mat([D, A_1, B])), D, A_1])
	"""

def inverse_woodburry_id(alpha, B, c, D):
	# when A_1 = alpha*Id it is much faster
	C_1 = np.diag([1./c_i for c_i in c])
	Z_1 = lin.inv(C_1 + alpha*np.dot(D,B))
	Q = (alpha**2)*mul_mat([B, Z_1, D])
	return alpha*np.diag([1]*Q.shape[0]) - Q

def woodburry_nystrom(n, matrices, coeffs, alpha=1e10):

	C, v, s_1, u = matrices[0]
	B_i = np.dot(C, np.transpose(v))
	D_i = np.dot(np.transpose(u), np.transpose(C))
	return inverse_woodburry_id(alpha, B_i, s_1, D_i)

	"""
	A_1 = alpha*np.diag(np.ones((n,), dtype=np.float64))
	for mat, coeff in zip(matrices, coeffs):
		C, v, s_1, u = mat
		B_i = np.dot(C, np.transpose(v))
		D_i = np.dot(np.transpose(u), np.transpose(C))

		A_1 = inverse_woodburry(A_1, B_i, s_1, D_i)
	return A_1
	"""

def nystrom(M, m):
	if M.ndim !=2:
		raise Exception("wrong number of dimensions, expected 2 got %i", M.ndim)
	if M.shape[0] != M.shape[1]:
		raise Exception("Expected square matrix, got dimension (%i, %i)", M.shape[0], M.shape[1])
	if m > M.shape[0] or m <0:
		raise Exception("subsampling dimension must be lower than the number of rows and positive but %i > %i", m, M.shape[0])
	# pick k the highest rank so that the matrix is invertible
	
	indices = np.random.permutation(M.shape[1])[:m]
	indices = np.sort(indices)
	# C Matrix :
	#C = np.zeros((M.shape[0], m), dtype=M.dtype)
	C = np.concatenate([M[:,index:index+1] for index in indices], axis=1)
	#W = np.zeros((m, m), dtype=M.dtype)
	W = np.concatenate([C[index:index+1,:] for index in indices], axis=0)
	"""
	for i in range(m):
		C[:,i]=M[:, indices[i]]
	"""
	# W matrix :
	"""
	for j in range(m):
		W[j,:] = C[indices[j], :]
	"""
	# in case W is non positive definite, we keep u and v cause they are not the transpose of one another
	[u, s, v] = lin.svd(W) # W = u*diag(s)*v
	# k is the rank of the matrices : we want s to have non zero diagonal elements so we reduce k=m if it is the case
	k = m
	while s[k-1]==0:
		k-=1
	s = s[:k]
	u = u[:, :k]; v=v[:k, :]
	s_1= np.copy(s)
	# garder les plus grands valeurs propres pour s_1 !!!!!!
	for i in range(k):
		s_1[i] = 1./s[i]

	#Z = M - mul_mat([C, np.transpose(v), np.diag(s_1), np.transpose(u), np.transpose(C)])
	#print lin.norm(Z, 'fro')
	return (C, v, s_1, u) #mul_mat([C, np.transpose(v), np.diag(s_1), np.transpose(u), np.transpose(C)])


def ensemble_nystrom(M, m, epochs):

	S_tilde = []
	for epoch in range(epochs):
		S_tilde.append(nystrom(M,m))
	# mean error
	#M = lin.inv(M)
	# matching pursuit alternative
	#matrices = [ mul_mat([C, np.transpose(v), np.diag(s_1), np.transpose(u), np.transpose(C)]) for (C, v, s_1, u) in S_tilde]
	#coeffs = mp(M, matrices)
	coeffs = [1./len(S_tilde)]*len(S_tilde)
	return S_tilde, coeffs
	"""
	M_tilde = np.zeros_like(M, dtype=M.dtype)
	for mat, coeff in zip(matrices, coeffs):
		M_tilde += coeff*mat
	a = lin.norm(M - M_tilde, 'fro')

	M_tilde = np.zeros_like(M, dtype=M.dtype)
	n = len(M)
	#matrices = [ np.random.ranf((n,n)) for i in range(100)]
	coeff = 1./len(S_tilde)
	for mat in matrices:
		M_tilde += mat

	b =  lin.norm(M - M_tilde, 'fro')
	print (a,b)
	if a >b:
		print 'MERDE'
		import pdb
		pdb.set_trace()
	"""

def approximate_inverse(M, m, epochs=1):
	n = M.shape[0]
	if m == n:
		return lin.inv(M + 1e-5*np.diag([1]*n))
		epochs = 1
	#m = (int) (n*p)
	matrices, coeffs = ensemble_nystrom(M, m, epochs)
	return woodburry_nystrom(n, matrices, coeffs) # TO TEST !!!!
	
	
if __name__=="__main__":
	K = np.random.ranf((10, 10))
	m=2
	for test in range(200):
		for i in range(10):
			for j in range(10):
				if i<j:
					K[i,j]=K[j,i]
		ensemble_nystrom(K,m, m, 3)
