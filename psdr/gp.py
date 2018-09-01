import numpy as np
from scipy.spatial.distance import pdist, squareform
import scipy.linalg
from scipy.linalg import eigh, expm, logm
from scipy.optimize import fmin_l_bfgs_b
from itertools import product
from opt import check_gradient
from poly_ridge import LegendreTensorBasis

__all__ = [ 'fit_gp', 'GaussianProcess']



def fit_gp(X, y, rank = None, poly_degree = None, structure = 'tril', L0 = None, Lfixed = None,
		_check_gradient = False):
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

	Internally, we optimize with respect to the matrix log of the matrix L, namely,
		
		L(ell) = expm(ell).

	In the constant and diagonal cases, this corresponds to the standard practice of
	optimizing with respect to the (scalar) log of the scaling parameters.  In our experience,
	working with this logarithmic parameterization increases the accuracy of the derivatives.

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
	obj: float
		Log-likelihood objective function value

	References
	----------
	[RW06] Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I. Williams,
		2006 MIT Press
	[Jon01] A Taxonomy of Global Optimization Methods Based on Response Surfaces,
		Donald R. Jones, Journal of Global Optimization, 21, pp. 345--383, 2001.
	[MN10] "The complex step approximation to the Frechet derivative of a matrix function", 
		Awad H. Al-Mohy and Nicholas J. Higham, Numerical Algorithms, 2010 (53), pp. 133--148.
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
		tril_flat = np.array([ i*m + j for i,j in tril_ij])

		# Construct starting lower-triangular matrix
		if L0 is None:
			ell0 = np.random.randn(len(tril_ij))
		else:
			ell0 = np.array([L0[i,j] for i,j in tril_ij])

		def make_L(ell):
			# Construct the L matrix	
			L = np.zeros((m*m,), dtype = ell.dtype)
			L[tril_flat] = ell
			L = L.reshape(m,m)
			return scipy.linalg.expm(L) - np.eye(m)

	elif structure == 'diag':
		tril_ij = [ (i,i) for i in range(m)]
		# Construct starting diagonal matrix	
		if L0 is None:
			ell0 = np.random.randn(m)
		else:
			ell0 = np.array([L0[i,j] for i,j in tril_ij])
	
		def make_L(ell):
			return np.diag(np.exp(ell))	

	elif structure == 'const':
		# Construct starting constant
		if L0 is None:
			ell0 = np.random.randn(1)
		else:
			ell0 = L0[0,0].reshape(1)

		def make_L(ell):
			return np.exp(ell)*np.eye(m)

	elif structure == 'scalar_mult':
		assert Lfixed is not None, "In order to use structure = 'scalar_mult', Lfixed must be provided"
		
		# Construct starting constant
		if L0 is None:
			ell0 = np.random.randn(1)
		else:
			ell0 = L0[0,0].reshape(1)

		def make_L(ell):
			return np.exp(ell)*Lfixed

	# Construct the basis for the polynomial part
	if poly_degree is None:
		V = np.zeros((M,0))
	else:
		basis = LegendreTensorBasis(m, poly_degree)
		V = basis.V(X)  


	def log_marginal_likelihood(ell, return_obj = True, return_grad = False, return_alpha_beta = False):
		L = make_L(ell)
		# Compute the squared distance
		Y = np.dot(L, X.T).T
		dij = pdist(Y, 'sqeuclidean') 

		# Covariance matrix
		K = np.exp(-0.5*squareform(dij))
		ew, ev = eigh(K)
		if return_grad:
			print "ew", ew
		# This is so we don't have trouble computing the objective function
		ew = np.maximum(ew, 5*np.finfo(float).eps)

		# Solve the linear system to compute the coefficients
		#alpha = np.dot(ev,(1./(ew+tikh))*np.dot(ev.T,y))
		A = np.vstack([np.hstack([K, V]), np.hstack([V.T, np.zeros((V.shape[1], V.shape[1]))])])
		b = np.hstack([y, np.zeros(V.shape[1])])

		# As A can be singular, we use an eigendecomposition based inverse
		ewA, evA = eigh(A)
		I = (np.abs(ewA) > 5*np.sqrt(np.finfo(float).eps))
		x = np.dot(evA[:,I],(1./ewA[I])*np.dot(evA[:,I].T,b))

		alpha = x[:M]
		beta = x[M:]

		if return_alpha_beta:
			return alpha, beta

		if return_obj:
			# Should this be with yhat or y?
			# yhat = y - np.dot(V, beta)
			# Doesn't matter because alpha in nullspace of V.T
			# RW06: (5.8)
			obj = 0.5*np.dot(y, alpha) + 0.5*np.sum(np.log(ew))
			if not return_grad:
				return obj
		
		# Now compute the gradient
		# Build the derivative of the covariance matrix K wrt to L

		if structure == 'tril':
			dK = np.zeros((M,M, len(ell)))
			for idx, (k, el) in enumerate(tril_ij):
				eidx = np.zeros(ell.shape)
				eidx[idx] = 1.
				# Approximation of the matrix exponential derivative [MH10]
				h = 1e-10
				dL = np.imag(make_L(ell + 1j*h*eidx))/h
				print "dL", dL
				dY = np.dot(dL, X.T).T
				for i in range(M):
					# Evaluate the dot product
					# dK[i,j,idx] -= np.dot(Y[i] - Y[j], dY[i] - dY[j])
					dK[i,:,idx] -= K[i,:]*np.sum((Y[i] - Y)*(dY[i] - dY), axis = 1)
			#for idx in range(len(tril_ij)):
			#	dK[:,:,idx] *= K
	
		elif structure == 'diag':
			dK = np.zeros((M,M, len(ell)))
			for idx, (k, el) in enumerate(tril_ij):
				for i in range(M):
					dK[i,:,idx] -= (Y[i,k] - Y[:,k])*(Y[i,el] - Y[:,el])
			for idx in range(len(tril_ij)):
				dK[:,:,idx] *= K	

		elif structure in ['const', 'scalar_mult']:
			# In the scalar case everything drops and we 
			# simply need 
			# dK[i,j,1] = (Y[i] - Y[j])*(Y[i] - Y[j])*K
			# which we have already computed
			dK = -(squareform(dij)*K).reshape(M,M,1)
				

		# Now compute the gradient
		grad = np.zeros(len(ell))
		print "alpha", alpha
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


	if _check_gradient:
		print make_L(ell0)
		grad = log_marginal_likelihood(ell0, return_grad = True, return_obj = False)
		err = check_gradient(log_marginal_likelihood, ell0, grad, verbose = True)
		return err

	grad = lambda x: log_marginal_likelihood(x, return_obj = False, return_grad = True)
	ell, obj, d = fmin_l_bfgs_b(log_marginal_likelihood, ell0, fprime = grad, disp = True)
	L = make_L(ell)
	alpha, beta = log_marginal_likelihood(ell, return_alpha_beta = True)

	return L, alpha, beta, obj
	
	
class GaussianProcess(object):
	def __init__(self, rank = None):
		self.rank = rank

	def fit(self, X, y):
		""" Fit a Gaussian process model

		"""
		
		fit_gp(X, y, rank = self.rank)
		pass		



if __name__ == '__main__':
	m = 3
	#np.random.seed(0)
	X = np.random.randn(50,m)
	a = np.ones(m)
	y = np.dot(a.T, X.T).T + 1
	A = np.random.randn(m,m)
	Q, R = np.linalg.qr(A)
	Lfixed = R.T
	#L0 = 5e-1*Lfixed
	L0 = np.zeros((m,m))
	L0[-1,:] = np.random.randn(m)
	print scipy.linalg.expm(L0)
	X = np.dot(L0, L0)
	for i in range(10):
		print X
		X = np.dot(L0, X)
	#L, alpha, beta, obj = fit_gp(X, y, structure = 'tril', L0 = L0, rank = 1)
	#print L
	#print alpha
	#print beta
	#print obj

	
