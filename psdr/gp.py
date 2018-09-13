import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.linalg
from scipy.linalg import eigh, expm, logm
from scipy.optimize import fmin_l_bfgs_b
from itertools import product
from opt import check_gradient
from poly_ridge import LegendreTensorBasis

__all__ = [ 'fit_gp', 'GaussianProcess']

def fit_gp(X, y, rank = None, poly_degree = None, structure = 'tril', L0 = None, Lfixed = None,
		_check_gradient = False, verbose = False):
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
	if rank is not None:
		print "Low rank estimates currently have numerical instability in gradient; please avoid"

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
		
		# This is so we don't have trouble computing the objective function
		ew = np.maximum(ew, 5*np.finfo(float).eps)

		# Solve the linear system to compute the coefficients
		#alpha = np.dot(ev,(1./(ew+tikh))*np.dot(ev.T,y))
		A = np.vstack([np.hstack([K, V]), np.hstack([V.T, np.zeros((V.shape[1], V.shape[1]))])])
		b = np.hstack([y, np.zeros(V.shape[1])])

		# As A can be singular, we use an eigendecomposition based inverse
		ewA, evA = eigh(A)
		I = (np.abs(ewA) > 5*np.finfo(float).eps)
		#I = (ewA > 0)
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
					dK[i,:,idx] -= np.sum((Y[i] - Y)*(dY[i] - dY), axis = 1)
			for idx in range(len(tril_ij)):
				dK[:,:,idx] *= K
	
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
	
		for k in range(len(ell)):
			#Kinv_dK = np.dot(ev, np.dot(np.diag(1./(ew+tikh)),np.dot(ev.T,dK[:,:,k])))
			#I = (ew > 0.1*np.sqrt(np.finfo(float).eps))
			I = (ew>5*np.finfo(float).eps)
			Kinv_dK = np.dot(ev[:,I], (np.dot(ev[:,I].T,dK[:,:,k]).T/ew[I]).T)
			# Note flipped signs from RW06 eq. 5.9
			grad[k] = 0.5*np.trace(Kinv_dK)
			grad[k] -= 0.5*np.dot(alpha, np.dot(alpha, dK[:,:,k]))
		
		if return_obj and return_grad:
			return obj, grad
		if not return_obj:
			return grad


	if _check_gradient:
		grad = log_marginal_likelihood(ell0, return_grad = True, return_obj = False)
		err = check_gradient(log_marginal_likelihood, ell0, grad, verbose = True)
		return err

	grad = lambda x: log_marginal_likelihood(x, return_obj = False, return_grad = True)
	ell, obj, d = fmin_l_bfgs_b(log_marginal_likelihood, ell0, fprime = grad, disp = verbose)
	L = make_L(ell)
	alpha, beta = log_marginal_likelihood(ell, return_alpha_beta = True)
	# TODO: Sometimes this returns outrageously large, negative results, indicative of a bug
	return L, alpha, beta, obj
	
	
class GaussianProcess(object):
	def __init__(self, structure = 'const', n_init = 1, rank = None, poly_degree = None, L0s = None, **kwargs):
		self.structure = structure
		self.rank = rank
		self.kwargs = kwargs
		self.n_init = n_init
		self.poly_degree = poly_degree
		self.L0s = L0s

		self._best_score = np.inf

	def fit(self, X, y):
		""" Fit a Gaussian process model

		"""
		self.X = np.copy(X)
		self.y = np.copy(y)
		self.refine(X, y, 	n_init = self.n_init, L0s = self.L0s)	

	def refine(self, X, y, n_init = 1, L0s = None):
		if L0s is None:
			L0s = [ np.random.randn(X.shape[1], X.shape[1]) for i in range(n_init)]
		
		res = [ fit_gp(X, y, rank = self.rank, structure = self.structure, L0 = L0, poly_degree = self.poly_degree, **self.kwargs) for L0 in L0s ]
		# Remove excessively small solutions
		res = [res_i for res_i in res if np.max(np.abs(res_i[0])) > 1e-10]
		#print [res_i[-1] for res_i in res]
		#print [res_i[0] for res_i in res]
		k = np.argmin([res_i[-1] for res_i in res])
		if res[k][-1] < self._best_score:
			self.L = res[k][0]
			self.alpha = res[k][1]
			self.beta = res[k][2]
			self._best_score = res[k][3]

	def predict(self, Xnew, return_cov = False):
		Y = np.dot(self.L, self.X.T).T
		Ynew = np.dot(self.L, Xnew.T).T
		dij = cdist(Ynew, Y, 'sqeuclidean')	
		K = np.exp(-0.5*dij)
		if self.poly_degree is not None:
			basis = LegendreTensorBasis(self.X.shape[1], self.poly_degree)
			V = basis.V(Xnew)
		else:
			V = np.zeros((Xnew.shape[0],0))

		fXnew = np.dot(K, self.alpha) + np.dot(V, self.beta)
		if return_cov:
			KK = np.exp(-0.5*squareform(pdist(Y, 'sqeuclidean')))
			ew, ev = eigh(KK)	
			I = (ew > 500*np.finfo(float).eps)
			z = ev[:,I].dot( np.diag(1./ew[I]).dot(ev[:,I].T.dot(K.T)))
			#z = np.dot(ev[:,I], np.dot(np.diag(1./ew[I]).dot(ev[:,I].T.dot( K.T)) ))
			cov = np.array([ 1 - np.dot(K[i,:], z[:,i]) for i in range(Xnew.shape[0])])
			cov[cov< 0] = 0.
			return fXnew, cov
		else:
			return fXnew

if __name__ == '__main__':
	m = 2
	#np.random.seed(0)
	X = np.random.randn(100,m)
	#Xnew = np.linspace(-1,1,100).reshape(-1,1)
	Xnew = np.random.randn(10,m)
	a = 2*np.ones(m)
	y = np.dot(a.T, X.T).T**2 + 1

	gp = GaussianProcess(poly_degree = None, n_init = 10, structure = 'const')
	gp.fit(X, y)
	yhat, cov = gp.predict(Xnew, return_cov = True)
	print y
	print yhat
	print cov
	#print y - gp.predict(X)
	#print gp.beta
	#print gp.L	
