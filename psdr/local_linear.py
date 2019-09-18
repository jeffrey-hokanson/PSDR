""" Local linear models for use in other functions
"""
import numpy as np
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache


@lru_cache(maxsize = int(10))
def _compute_p1(M, perplexity):
	r""" The constant appearing in VC13, eq. 9
	"""
	res = root_scalar(lambda x: np.log(min(np.sqrt(2*M), perplexity)) - 2*(1 - x)*np.log(M/(2*(1 - x))), bracket = [3./4,1 - 1e-14],)
	return res.root


def log_entropy(beta, d):
	p = np.exp(-beta*d)
	sum_p = np.sum(p)
	# Shannon entropy H = np.sum(-p*np.log2(p)) 
	# More stable formula 
	return beta*np.sum(p*d/sum_p) + np.log(sum_p)

def perplexity_bandwidth(d, perplexity):
	r"""Compute the 
	"""
	M = len(d)
	p1 = _compute_p1(M, perplexity)
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
	
	log_perplexity = np.log(perplexity)

	# Compute bandwidth beta
	res = root_scalar(lambda beta: log_entropy(beta, d) - log_perplexity,
		bracket = [beta1, beta2],
		#x0 = sum([beta1, beta2])/2,
		method = 'brenth',
		#method = 'newton',
		#fprime = log_entropy_der,
		rtol = 1e-10)
	beta = res.root
	return beta


def local_linear_grad(X, fX, perplexity = None, bandwidth = None, Xt = None):
	r""" Estimate gradients using localized linear models

	Parameters
	----------
	X: array-like (M, m)
		Places where the function is evaluated
	fX: array-like (M,)
		Value of the function at those locations
	perplexity: None, int, or False
		If None, defaults to m+1.
	bandwidth: None or positive float
		If specified, set the 	
	"""

	M, m = X.shape
	fX = fX.flatten()
	assert len(fX) == M, "Number of function evaluations does not match number of samples"

	if Xt is None:
		Xt = X

	if perplexity is None and bandwidth is None:
		perplexity = m+1
	elif perplexity is None and bandwidth is not None:
		perplexity = False

	if perplexity is not False:
		assert perplexity >= 2 and perplexity < M, "Perplexity must be in the interval [2,M)"
		p1 = _compute_p1(M, perplexity) 
	else:
		if bandwidth is None:
			# Bandwidth from Xia 2007, [Li18, eq. 11.5] 
			bandwidth = 2.34*M**(-1./(max(Z.shape[1], 3) +6))


	# Storage for the gradients
	grads = np.zeros((len(Xt), m))
	Y = np.hstack([np.ones((M, 1)), X])

	for i, xi in enumerate(Xt):
		d = cdist(X, xi.reshape(1,-1), 'sqeuclidean').flatten()
		if perplexity:
			beta = perplexity_bandwidth(d, perplexity)
		else:
			beta = bandwidth

		try:
			# Weights associated with each point
			sqrt_weights = np.exp(-0.5*beta*d).reshape(-1,1)
			g, _, _, _ = np.linalg.lstsq(sqrt_weights*Y, sqrt_weights*fX.reshape(-1,1), rcond = None)
			g = g.flatten()
		except np.linalg.LinAlgError:
			g = np.zeros(m+1)

		grads[i] = g[1:]

	return grads

