from __future__ import print_function
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import scipy.linalg
from scipy.linalg import eigh, expm, logm
from scipy.optimize import fmin_l_bfgs_b
import scipy.optimize
from itertools import product
#from opt import check_gradient
from .basis import LegendreTensorBasis
from .function import BaseFunction
__all__ = ['GaussianProcess']

	
	
class GaussianProcess(BaseFunction):
	r""" Fits a Gaussian Process by maximizing the marginal likelihood

	Given :math:`M` pairs of :math:`\mathbf{x}_i \in \mathbb{R}^m` and :math:`y_i \in \mathbb{R}`,
	this fits a model 

	.. math::

		g(\mathbf{x}) = 
			\sum_{i=1}^M \alpha_i e^{-\| \mathbf{L}( \mathbf{x} - \mathbf{x}_i) \|_2^2} 
			+ \sum_{j} \beta_j \psi_j(x)

	where :math:`\mathbf{L}` is a lower triangular matrix,
	:math:`\lbrace \psi_j \rbrace_j` is a polynomial basis (e.g., linear),
	and :math:`\boldsymbol{\alpha}, \boldsymbol{\beta}` are vectors of weights.
	These parameters are choosen to maximize the log-marginal likelihood (see [RW06]_, Sec. 2.7), 
	or equivalently minimize

	.. math::

		\min_{\mathbf{L}} & \ 
			\frac12 \mathbf{y}^\top \boldsymbol{\alpha}
			+ \frac12 \log \det (\mathbf{K} + \tau \mathbf{I}), & \quad 
			[\mathbf{K}]_{i,j} &= e^{-\frac12 \| \mathbf{L}(\mathbf{x}_i -\mathbf{x}_j)\|_2^2} \\
		\text{where} & \ 
			\begin{bmatrix}
				\mathbf{K} + \tau \mathbf{I} & \mathbf{V} \\ \mathbf{V}^\top & \mathbf{0}
			\end{bmatrix} 
			\begin{bmatrix}
				\boldsymbol{\alpha} \\
				\boldsymbol{\beta} 
			\end{bmatrix}
			= 
			\begin{bmatrix}
				\mathbf{y} \\ \mathbf{0}
			\end{bmatrix},
			&
			\quad [\mathbf{V}]_{i,j} &= \psi_j(\mathbf{x}_i).
	
	Here :math:`\tau` denotes the nugget--a Tikhonov-like regularization term commonly used to 
	combat the ill-conditioning of the kernel matrix :math:`\mathbf{K}`.
	The additional constraint for the polynomial basis is described by Jones
	([Jon01]_ eq. (3) and (4)).


	This class allows for four kinds of distance matrices :math:`\mathbf{L}`:

	================ 	=============================================
	scalar   			:math:`\mathbf{L} = \ell \mathbf{I}`
	scalar multiple 	:math:`\mathbf{L} = \ell \mathbf{L}_0`
	diagonal 			:math:`\mathbf{L} = \text{diag}(\boldsymbol{\ell})` 
	lower triangular	:math:`\mathbf{L}` is lower triangular
	================ 	=============================================
		
	Internally, we optimize with respect to the matrix log;
	specificially, we parameterize :math:`\mathbf{L}` in terms of
	some scalar or vector :math:`\boldsymbol{\ell}`; i.e., 
	

	================ 	=============================================
	scalar   			:math:`\mathbf{L}(\ell) = e^{\ell}\mathbf{I}`
	scalar multiple 	:math:`\mathbf{L}(\ell) = e^{\ell} \mathbf{L}_0`
	diagonal 			:math:`\mathbf{L}(\ell) = \text{diag}(\lbrace e^{\ell_i} \rbrace_{i})` 
	lower triangular	:math:`\mathbf{L}(\ell) = e^{\text{tril}(\boldsymbol{\ell})}`
	================ 	=============================================

	In the constant and diagonal cases, this corresponds to the standard practice of
	optimizing with respect to the (scalar) log of the scaling parameters. 
	Our experience suggests that the matrix log similarly increases the accuracy
	when working with the lower triangular parameterization.  
	For the first three classes we have simple expressions for the derivatives,
	but for the lower triangular parameterization we use complex step
	approximation to compute the derivative [MN10]_. 
	With this derivative information,
	we solve the optimization problem using L-BFGS as implemented in scipy.
	



	Parameters
	----------
	structure: ['tril', 'diag', 'const', 'scalar_mult']
		Structure of the matrix L, either

		* const: constant * eye
		* scalar_mult
		* tril: lower triangular
		* diag: diagonal	



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
	.. [RW06] Gaussian Processes for Machine Learning, Carl Edward Rasmussen and Christopher K. I. Williams,
		2006 MIT Press

	.. [Jon01] A Taxonomy of Global Optimization Methods Based on Response Surfaces,
		Donald R. Jones, Journal of Global Optimization, 21, pp. 345--383, 2001.

	.. [MN10] "The complex step approximation to the Frechet derivative of a matrix function", 
		Awad H. Al-Mohy and Nicholas J. Higham, Numerical Algorithms, 2010 (53), pp. 133--148.
	"""
	def __init__(self, structure = 'const', degree = None, nugget = None, Lfixed = None,
		n_init = 1):
		self.structure = structure
		
		self.rank = None # currently disabled due to conditioning issues
		self.n_init = n_init
		self.degree = degree

		self._best_score = np.inf

		if nugget is None:
			nugget = 5*np.finfo(float).eps
		self.nugget = nugget

		if structure is 'scalar_mult':
			assert Lfixed is not None, "Must specify 'Lfixed' to use scalar_mult"
			self.Lfixed = Lfixed

	def _make_L(self, ell):
		r""" Constructs the L matrix from the parameterization corresponding to the structure
		"""


		if self.structure is 'const':
			return np.exp(ell)*np.eye(self.m)
		elif self.structure is 'scalar_mult':
			return np.exp(ell)*self.Lfixed
		elif self.structure is 'diag':
			return np.diag(np.exp(ell))	
		elif self.structure is 'tril':
			# Construct the L matrix	
			L = np.zeros((self.m*self.m,), dtype = ell.dtype)
			L[self.tril_flat] = ell
			L = L.reshape(self.m,self.m)
			
			# This is a more numerically stable way to compute expm(L) - I
			#Lexp = L.dot(scipy.linalg.expm(L))
			
			# JMH 8 Aug 2019: I'm less sure about the value of parameterizing as L*expm(L)
			# Specifically, having the zero matrix easily accessible doesn't seem like a good thing
			# and he gradient is more accurately computed using this form. 
			Lexp = scipy.linalg.expm(L)
			
			return Lexp

	def _log_marginal_likelihood(self, ell, X = None, y = None, return_obj = True, return_grad = False, return_alpha_beta = False):
		
		if X is None: X = self.X
		if y is None: y = self.y 

		# Extract basic constants
		M = X.shape[0]	
		m = X.shape[1]	


		L = self._make_L(ell)
		# Compute the squared distance
		Y = np.dot(L, X.T).T
		dij = pdist(Y, 'sqeuclidean') 
		
		# Covariance matrix
		K = np.exp(-0.5*squareform(dij))

		# Solve the linear system to compute the coefficients
		#alpha = np.dot(ev,(1./(ew+tikh))*np.dot(ev.T,y)) 
		A = np.vstack([np.hstack([K + self.nugget*np.eye(K.shape[0]), self.V]), 
					   np.hstack([self.V.T, np.zeros((self.V.shape[1], self.V.shape[1]))])])
		b = np.hstack([y, np.zeros(self.V.shape[1])])

		# As A can be singular, we use an eigendecomposition based inverse
		ewA, evA = eigh(A)
		I = (np.abs(ewA) > 5*np.finfo(float).eps)
		x = np.dot(evA[:,I],(1./ewA[I])*np.dot(evA[:,I].T,b))

		alpha = x[:M]
		beta = x[M:]

		if return_alpha_beta:
			return alpha, beta

		ew, ev = scipy.linalg.eigh(K + self.nugget*np.eye(K.shape[0]))
		#if np.min(ew) <= 0:
		#	bonus_regularization = -2*np.min(ew)+1e-14 
		#	ew += bonus_regularization
		#	K += bonus_regularization*np.eye(K.shape[0])

		if return_obj:
			# Should this be with yhat or y?
			# yhat = y - np.dot(V, beta)
			# Doesn't matter because alpha in nullspace of V.T
			# RW06: (5.8)
			with np.errstate(invalid = 'ignore'):
				obj = 0.5*np.dot(y, alpha) + 0.5*np.sum(np.log(ew))
			if not return_grad:
				return obj
		
		# Now compute the gradient
		# Build the derivative of the covariance matrix K wrt to L

		if self.structure == 'tril':
			dK = np.zeros((M,M, len(ell)))
			for idx, (k, el) in enumerate(self.tril_ij):
				eidx = np.zeros(ell.shape)
				eidx[idx] = 1.
				# Approximation of the matrix exponential derivative [MH10]
				h = 1e-10
				dL = np.imag(self._make_L(ell + 1j*h*eidx))/h
				dY = np.dot(dL, X.T).T
				for i in range(M):
					# Evaluate the dot product
					# dK[i,j,idx] -= np.dot(Y[i] - Y[j], dY[i] - dY[j])
					dK[i,:,idx] -= np.sum((Y[i] - Y)*(dY[i] - dY), axis = 1)
			for idx in range(len(self.tril_ij)):
				dK[:,:,idx] *= K
	
		elif self.structure == 'diag':
			dK = np.zeros((M,M, len(ell)))
			for idx, (k, el) in enumerate(self.tril_ij):
				for i in range(M):
					dK[i,:,idx] -= (Y[i,k] - Y[:,k])*(Y[i,el] - Y[:,el])
			for idx in range(len(self.tril_ij)):
				dK[:,:,idx] *= K	

		elif self.structure in ['const', 'scalar_mult']:
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
			#I = (ew>5*np.finfo(float).eps)
			#print "k", k, "dK", dK.shape

			Kinv_dK = np.dot(ev, (np.dot(ev.T,dK[:,:,k]).T/ew).T)
			# Note flipped signs from RW06 eq. 5.9
			grad[k] = 0.5*np.trace(Kinv_dK)
			grad[k] -= 0.5*np.dot(alpha, np.dot(alpha, dK[:,:,k]))

		if return_obj and return_grad:
			return obj, grad
		if not return_obj:
			return grad


	def _obj(self, ell, X = None, y = None):
		return self._log_marginal_likelihood(ell, X, y, 
			return_obj = True, return_grad = False, return_alpha_beta = False)
	
	def _grad(self, ell, X = None, y = None):
		return self._log_marginal_likelihood(ell, X, y, 
			return_obj = False, return_grad = True, return_alpha_beta = False)

	
	def _fit_init(self, X, y):
		m = self.m = X.shape[1]
		self.X = X
		self.y = y
		# Setup structure based properties
		if self.structure == 'tril':
			if self.rank is None: rank = m
			else: rank = self.rank
			
			self.tril_ij = [ (i,j) for i, j in zip(*np.tril_indices(m)) if i >= (m - rank)]
			self.tril_flat = np.array([ i*m + j for i,j in self.tril_ij])

		elif self.structure == 'diag':
			self.tril_ij = [ (i,i) for i in range(m)]


		# Cache Vandermonde matrix on sample points
		if self.degree is not None:
			self.basis = LegendreTensorBasis(self.degree, X = X)
			self.V = self.basis.V(X)
		else:
			self.V = np.zeros((X.shape[0],0))
		


	def fit(self, X, y, L0 = None):
		""" Fit a Gaussian process model

		Parameters
		----------
		X: array-like (M, m)
			M input coordinates of dimension m
		y: array-like (M,)
			y[i] is the output at X[i]
		"""
		X = np.array(X)
		y = np.array(y).flatten()
	
		# Initialized cached values for fit
		self._fit_init(X, y)	

	
		if L0 is None:
			L0 = np.eye(self.m)

		if self.structure == 'tril':
			ell0 = np.array([L0[i,j] for i, j in self.tril_ij])
		elif self.structure == 'diag':
			if len(L0.shape) == 1:
				ell0 = L0.flatten()
			else:
				ell0 = np.array([L0[i,i] for i, j in self.tril_ij])
		elif self.structure == 'scalar_mult':
			ell0 = np.array(L0.flatten()[0])
		elif self.structure == 'const':
			ell0 = np.array(L0.flatten()[0])

		# Actually do the fitting
		# TODO: Implement multiple initializations
		self._fit(ell0)


	def _fit(self, ell0):
		# the implementation in l_bfgs_b seems flaky when we have invalid values
		#ell, obj, d = fmin_l_bfgs_b(self._obj, ell0, fprime = self._grad, disp = True)
		res = scipy.optimize.minimize(self._obj, 
				ell0, 
				jac = self._grad,
				#method = 'L-BFGS-B',
				#options = {'disp': True}, 
			)
		ell = res.x
		self.L = self._make_L(ell)
		self.alpha, self.beta = self._log_marginal_likelihood(ell, 
			return_obj = False, return_grad = False, return_alpha_beta = True)

	def eval(self, Xnew, return_cov = False):
		Y = np.dot(self.L, self.X.T).T
		Ynew = np.dot(self.L, Xnew.T).T
		dij = cdist(Ynew, Y, 'sqeuclidean')	
		K = np.exp(-0.5*dij)
		if self.degree is not None:
			V = self.basis.V(Xnew)
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

