
from __future__ import print_function, division
import numpy as np
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist
from .opg import OuterProductGradient
from .subspace import ActiveSubspace
__all__ = ['PerplexityOuterProductGradient']


def perplexity_opg_grads(X, fX, perplexity = None):
	r""" Compute the gradients in Outer Product Gradient using a variable bandwidth
	"""
	################################################################################	
	# Compute gradient estimates from samples using a technique similar to
	# outer-product gradient (OPG)
	################################################################################	
	M = X.shape[0]	
	m = X.shape[1]

	Y = np.hstack([np.ones((M, 1)), X])

	if perplexity is None:
		perplexity = min((m+1), M)
	else:
		perplexity = float(perplexity)
		perplexity = max(1, perplexity)
		perplexity = min(M, perplexity)

	log_perplexity = np.log(perplexity)
	
	# Compute the constant from [VC13,(9)]
	res = root_scalar(lambda x: np.log(min(np.sqrt(2*M), perplexity)) - 2*(1 - x)*np.log(M/(2*(1 - x))), bracket = [3./4,1 - 1e-14],)
	p1 = res.root
	
	
	opg_grads = []
	for i, xi in enumerate(X):
		# Compute 2-norm distance between points
		d = cdist(X, xi.reshape(1,-1), 'sqeuclidean').flatten()
		
		# Compute the bandwidth for target perplexity
		def log_entropy(beta):
			beta = min(max(beta, beta1), beta2)
			p = np.exp(-d*beta)
			#p[i] = 0.
			sum_p = np.sum(p)
			# Shannon entropy
			#H = np.sum(-p*np.log2(p)) 
			# More stable formula 
			return beta*np.sum(p*d/sum_p) + np.log(sum_p)

		def log_entropy_der(beta):
			beta = min(max(beta, beta1), beta2)
			p = np.exp(-d*beta)
			sum_p = np.sum(p)
			#p[i] = 0.
			# [VC13, eq. 4]
			return -beta*(np.sum(p*d**2/sum_p) - np.sum(p*d/sum_p)**2)

		# Compute upper and lower bounds of beta from [VC13, eq. (7) (8)]
		# These are constants appearing the bounds
		dM = np.max(d)
		d1 = np.min(d[d>0])
		delta2 = d - d1
		delta2 = np.min(delta2[delta2>0])
		deltaM = dM - d1

		# lower bound (7)
		beta1 = max(M*np.log(M/perplexity)/((M-1)*deltaM), np.sqrt(np.log(M/perplexity)/(dM**4 - d1**4)))
		# upper bound (8)
		beta2 = 1/delta2*np.log(p1/(1-p1)*(M - 1))
		
		# Compute bandwidth beta
		res = root_scalar(lambda beta: log_entropy(beta) - log_perplexity,
			bracket = [beta1, beta2],
			#x0 = sum([beta1, beta2])/2,
			method = 'brenth',
			#method = 'newton',
			#fprime = log_entropy_der,
			rtol = 1e-4)
		beta = res.root

		# Weights associated with each point
		weights = np.exp(-beta*d).reshape(-1,1)
		
		# This is sum_j weight_j * y_j y_j^T 
		A = Y.T.dot(weights*Y)
		# This is sum_j weight_j * y_j * fX_j
		b = np.sum((weights*fX.reshape(-1,1))*Y, axis = 0)
		# Estimate the coefficients of the line
		g = np.linalg.solve(A, b)
		# Extract the slope as the gradient
		opg_grads.append(g[1:])

	opg_grads = np.vstack(opg_grads)
	return opg_grads

class PerplexityOuterProductGradient(OuterProductGradient):
	r"""
	"""
	def __init__(self, perplexity = None):
		self.perplexity = perplexity

	def __str__(self):
		return "<Perplexity Outer Product Gradient>"

	def fit(self, X, fX):
		X = np.atleast_2d(X)
		fX = np.atleast_1d(fX)

		grads = perplexity_opg_grads(X, fX, perplexity = self.perplexity)
		ActiveSubspace.fit(self, grads)

