import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.linalg
from scipy.linalg import eigh
from scipy.optimize import fmin_l_bfgs_b
from itertools import product
from opt import check_gradient
from poly_ridge import LegendreTensorBasis

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
	is discussed in [Sec. 2.7,RW06] and implemented following [eq (3) & (4),Jon01].

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
	[Jon01] A Taxonomy of Global Optimization Methods Based on Resonse Surfaces,
		Donald R. Jones, Journal of Global Optimization, 21, pp. 345--383, 2001.

	"""
	m = X.shape[1]
	M = len(X)
	assert len(y) == M, "Number of samples must equal number of function values"
	assert structure in ['tril', 'diag', 'const', 'scalar_mult'], 'invalid structure parameter'

	if structure is not 'tril': assert rank is None, "Cannot specify rank unless structure='tril'"

	if structure == 'tril':
		if rank is None: rank = X.shape[1]
		# Indices for parameterizing the lower triangular part
		tril_ij = [ (i,j) for i, j in zip(*np.tril_indices(m)) if i >= (m - rank)]
		print tril_ij
		tril_flat = np.array([ i*m + j for i,j in tril_ij])

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
		tril_ij = [ (i,i) for i in range(m)]
		tril_flat = np.array([i for i in range(m)])
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
			ell0 = L0[0,0].reshape(1)

		def make_L(ell):
			return ell*np.eye(m)

	elif structure == 'scalar_mult':
		assert Lfixed is not None, "In order to use structure = 'scalar_mult', Lfixed must be provided"
		
		# Construct starting constant
		if L0 is None:
			ell0 = np.exp(np.random.randn(1))
		else:
			ell0 = L0[0,0].reshape(1)

		def make_L(ell):
			return ell*Lfixed

	# Construct the basis for the polynomial part
	if poly_degree is None:
		V = np.zeros((M,0))
	else:
		basis = LegendreTensorBasis(m, poly_degree)
		V = basis.V(X)  


	def log_marginal_likelihood(ell, return_obj = True, return_grad = False):
		L = make_L(ell)

		# Compute the squared distance
		Y = np.dot(L, X.T).T
		dij = pdist(Y, 'sqeuclidean') 

		# Covariance matrix
		K = np.exp(-0.5*squareform(dij))
		ew, ev = eigh(K)
		# This is so we don't have trouble computing the objective function
		ew = np.maximum(ew, 1*np.finfo(float).eps)

		# Solve the linear system to compute the coefficients
		#alpha = np.dot(ev,(1./(ew+tikh))*np.dot(ev.T,y))
		A = np.vstack([np.hstack([K, V]), np.hstack([V.T, np.zeros((V.shape[1], V.shape[1]))])])
		b = np.hstack([y, np.zeros(V.shape[1])])

		# As A can be singular, we use an eigendecomposition based inverse
		ewA, evA = eigh(A)
		I = (np.abs(ewA) > 5*np.finfo(float).eps)
		ewAinv = np.zeros(ewA.shape, ewA.dtype)
		ewAinv[I] = 1./ewA[I]
		x = np.dot(evA,ewAinv*np.dot(evA.T,b))

		alpha = x[:M]
		beta = x[M:]
		yhat = y - np.dot(V, beta)

		if return_obj:
			# Should this be with yhat or y?
			# RW06: (5.8)
			obj = 0.5*np.dot(yhat, alpha) + 0.5*np.sum(np.log(ew))
			if not return_grad:
				return obj
		
		# Now compute the gradient
		# Build the derivative of the covariance matrix K wrt to L

		if structure in ['tril', 'diag']:
			dK = np.zeros((M,M, len(ell)))
			#for i,j in product(range(M), range(M)):
			#	#print np.kron((Y[i] - Y[j]).reshape(-1,1), (X[i] - X[j]).reshape(1,-1)).shape
			#	gradij = np.kron(Y[i] - Y[j], X[i] - X[j])[tril_flat]
			#	dK[i,j,:] = -K[i,j]*gradij
			#for i, j in product(range(M), range(M)):
			#	for idx, (k, el) in enumerate(tril_ij):
			#		dK[i,j,idx] -= K[i,j]*(X[i,el] - X[j,el])*np.dot(L[k,:], X[i] - X[j])
			for i in range(M):
				for idx, (k, el) in enumerate(tril_ij):
					dK[i,:,idx] -= (X[i,el] - X[:,el])*np.dot(L[k,:], (X[i] - X).T).T		
			
			for idx in range(len(tril_ij)):
				dK[:,:,idx] *= K	
		elif structure in ['const', 'scalar_mult']:
			# We factor the ell[0] outside to take the derivative,
			# so we need to remove it from the distance matrix
			#dK = -ell[0]*(squareform(dij)/ell[0]**2*K).reshape(M,M,1)
			dK = -(1/ell[0])*(squareform(dij)*K).reshape(M,M,1)
				

		# Now compute the gradient
		grad = np.zeros(len(ell))
		for k in range(len(ell)):
			#Kinv_dK = np.dot(ev, np.dot(np.diag(1./(ew+tikh)),np.dot(ev.T,dK[:,:,k])))
			Kinv_dK = np.dot(ev, (np.dot(ev.T,dK[:,:,k]).T/ew).T)
			# Note flipped signs from RW06 eq. 5.9
			grad[k] = 0.5*np.trace(Kinv_dK)
			grad[k] -= 0.5*np.dot(alpha, np.dot(alpha, dK[:,:,k]))
			
		
		if return_obj and return_grad:
			return obj, grad
		if not return_obj:
			return grad


	print make_L(ell0)

	if True:
		grad = log_marginal_likelihood(ell0, return_grad = True, return_obj = False)
		print "gradient check", check_gradient(log_marginal_likelihood, ell0, grad, verbose = True)

#	grad = lambda x: log_marginal_likelihood(x, return_obj = False, return_grad = True)
#	ell, obj, d = fmin_l_bfgs_b(log_marginal_likelihood, ell0, fprime = grad, disp = True)
#	return make_L(ell)

	
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
	X = np.random.randn(10,m)
	a = np.ones(m)
	y = np.dot(a.T, X.T).T + 1
	A = np.random.randn(m,m)
	Q, R = np.linalg.qr(A)
	Lfixed = R.T
	L0 = np.zeros((m,m))
	L0[-1,:] = np.random.randn(m)
	L0 = 10*np.eye(m)
	#L0 = np.random.randn(m,m)
	print fit_gp(X, y, structure = 'tril', Lfixed = None, poly_degree = None, L0 = L0, rank = 2)
	print a

	
