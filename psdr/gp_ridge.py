import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.linalg import eigh
from scipy.optimize import fmin_l_bfgs_b
from itertools import product
from util import check_gradient


def fit_gp(X, y, rank = None, poly_degree = None, structure = 'tril', L0 = None, Lfixed = None):
	""" Fit a Gaussian Process by maximizing the marginal likelihood

	This code fits a model 
	
		y[i] = k(X[i]) alpha +  v(X[i]) beta

	where k(x) is the evaluation of the kernel (for a Gaussian process)
		
		k(x) = exp(-|| L( x - X[i]) ||^2_2)

	where L is a lower triangular matrix and v(x) corresponds to 
	a polynomial in a LegendreTensorBasis

		v(x) = [phi_0(x)  phi_1(x) ... phi_N(x)],
	
	where the inclusion of a specific (unknown) model for the mean 
	is discussed in [Sec. 2.7,RW06].

	The hyperparameters for kernel, namely the matrix L, are determined
	via maximizing the marginal likelihood following Algorithm 2.1 of RW06.


	Parameters
	----------
	X: np.ndarray(M, m)
		M input coordinates of dimension m
	y: np.ndarray(M)
		y[i] is the output at X[i]
	structure: ['tril', 'diag', 'const', 'scalar_mult']
		Structure of the matrix L, either
		* tril: lower triangular
		* diag: diagonal	
		* const: constant * eye
		* scalar_mult
	rank: int or None
		If structure is 'tril', this specifies the rank of L	


	Returns
	-------
	L: np.ndarray(m,m)
		Distance matrix
	alpha: np.ndarray(M)
		Weights for the Gaussian process kernel
	beta: np.ndarray(N)
		Weights for the polynomial component in the LegendreTensorBasis	


	Citations
	---------
	[RW06] Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I. Williams,
		2006 MIT Press

	"""
	m = X.shape[1]
	M = len(X)
	assert len(y) == M, "Number of samples must equal number of function values"
	assert structure in ['tril', 'diag', 'const', 'scalar_mult'], 'invalid structure parameter'

	if structure is not 'tril': assert rank is None, "Cannot specify rank unless structure='tril'"

	if structure == 'tril':
		# Indices for parameterizing the lower triangular part
		tril_ij = [ (i,j) for i, j in zip(*np.tril_indices(m)) if i >= (m - rank)]
		tril_flat = np.array([ i*m + j for i,j in tril_ij])
		tril_flat_trans = np.array([i + m*j for i,j in tril_ij])
	elif structure == 'scalar_mult':
		tril_ij = [ (i,j) for i, j in zip(*np.tril_indices(m,m)) ]
		tril_flat = np.array([ i*m + j for i,j in tril_ij])
		tril_flat_trans = np.array([i + m*j for i,j in tril_ij])

	else:
		tril_ij = [ (i,i) for i in range(m)]
		tril_flat = np.array([i for i in range(m)])
		tril_flat_trans = np.array([i + m*j for i,j in tril_ij])

	
	if structure == 'tril':
		if rank is None: rank = X.shape[1]

		# Construct starting lower-triangular matrix
		if L0 is None:
			ell0 = np.random.randn(len(tril_ij))
		else:
			ell0 = np.array([L0[i,j] for i,j in tril_ij])

		def make_L(ell):
			# Construct the L matrix	
			L = np.zeros((m*m,))
			L[tril_flat] = ell
			return L.reshape(m,m)

	elif structure == 'diag':
		# Construct starting diagonal matrix	
		if L0 is None:
			ell0 = np.exp(np.random.randn(m))
		else:
			ell0 = np.array([L0[i,j] for i,j in tril_ij])
	
		def make_L(ell):
			return np.diag(ell)	

	elif structure == 'const':
		# Construct starting constant
		if L0 is None:
			ell0 = np.exp(np.random.randn(1))
		else:
			ell0 = L0[0,0]

		def make_L(ell):
			return ell*np.eye(m)

	elif structure == 'scalar_mult':
		assert Lfixed is not None, "In order to use structure = 'scalar_mult', Lfixed must be provided"
		
		# Construct starting constant
		if L0 is None:
			ell0 = np.exp(np.random.randn(1))
		else:
			ell0 = L0[0,0]

		def make_L(ell):
			return ell*Lfixed


	def log_marginal_likelihood(ell, return_obj = True, return_grad = False, tikh = 0):
		L = make_L(ell)

		# Compute the squared distance
		Y = np.dot(L, X.T).T
		dij = pdist(Y, 'sqeuclidean') 

		# Covariance matrix
		K = np.exp(-0.5*squareform(dij))
		ew, ev = eigh(K)

		Kinv_y = np.dot(ev,(1./(ew+tikh))*np.dot(ev.T,y))
		if return_obj:
			obj = 0.5*np.dot(y, Kinv_y) + 0.5*np.sum(np.log(ew+tikh))
			if not return_grad:
				return obj
		
		# Now compute the gradient
		# Build the derivative of the covariance matrix K wrt to L
		dK = np.zeros((M,M, len(ell)))

		if structure in ['tril', 'diag']:
			for i,j in product(range(M), range(M)):
				gradij = np.kron(X[i] - X[j], Y[i] - Y[j])[tril_flat_trans]
				dK[i,j,:] = -K[i,j]*gradij
		elif structure == 'const':
			for i,j in product(range(M), range(M)):
				gradij = np.kron(X[i] - X[j], Y[i] - Y[j])[tril_flat_trans]
				dK[i,j,0] = -K[i,j]*np.sum(gradij)
		else:
			for i,j in product(range(M), range(M)):
				gradij = np.kron(X[i] - X[j], X[i] - X[j])[tril_flat_trans]
				dK[i,j,0] = -K[i,j]*np.sum(gradij)
				

		# Now compute the gradient
		grad = np.zeros(len(ell))
		for k in range(len(ell)):
			#Kinv_dK = np.dot(ev, np.dot(np.diag(1./(ew+tikh)),np.dot(ev.T,dK[:,:,k])))
			Kinv_dK = np.dot(ev, (np.dot(ev.T,dK[:,:,k]).T/(ew+tikh)).T)
			# Note flipped signs from RW06
			grad[k] = 0.5*np.trace(Kinv_dK)
			grad[k] -= 0.5*np.dot(Kinv_y, np.dot(Kinv_y, dK[:,:,k]))
			
		
		if return_obj and return_grad:
			return obj, grad
		if not return_obj:
			return grad


	if True:
		grad = log_marginal_likelihood(ell0, return_grad = True, return_obj = False)
		print "gradient check", check_gradient(log_marginal_likelihood, ell0, grad)

	grad = lambda x: log_marginal_likelihood(x, return_obj = False, return_grad = True)
	ell, obj, d = fmin_l_bfgs_b(log_marginal_likelihood, ell0, fprime = grad)
	return make_L(ell)

	
class GaussianProcessRidgeApproximation(object):
	def __init__(self, rank = None):
		self.rank = rank

	def fit(self, X, y):
		""" Fit a Gaussian process model

		"""
		
		fit_gp(X, y, rank = self.rank)
		pass		



if __name__ == '__main__':
	m = 5
	np.random.seed(0)
	X = np.random.randn(20,m)
	a = np.ones(m)
	y = np.dot(a.T, X.T).T
	A = np.random.randn(m,m)
	Q, R = np.linalg.qr(A)
	Lfixed = R.T
	print fit_gp(X, y, structure = 'scalar_mult', Lfixed = Lfixed)	


	
