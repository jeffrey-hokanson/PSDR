from __future__ import print_function, division
import numpy as np
from scipy.optimize import root_scalar
from scipy.spatial.distance import cdist
from .perplexity_opg import perplexity_opg_grads


def subspace_init(subspace_dimension, X = None, fX = None, grads = None):
	r""" Compute an inexpsive initial active subspace estimate 

	This computes an inexpensive estimate of the active subspace using 
	a combination of function samples and gradients.

	Parameters
	----------
	subspace_dimension: int	
		Number of dimensions in the active subspace to find
	X: array-like (M, m)
		Coordinates at which the function is evaluated
	fX: array-like (M,)
		Values of the function corresponding to X
	grads: array-like (M, m)
		Evaluations of the gradient of f

	References
	----------
	[VC13]  Max Vladymyrov and Miguel A.Carreira-Perpinan
		Entroptic Affninities: Properties and Efficient Numerical Computation.
		30th International Conference on Machine Learning, pp 1514--1522, 2013.
		http://proceedings.mlr.press/v28/vladymyrov13.pdf
	
	"""
	if grads is None and (X is None or fX is None):
		raise AssertionError("Either 'grads' or 'X' and 'fX' must be passed as arguments to this function")

	################################################################################	
	# Input standardization
	################################################################################	

	if grads is not None:
		grads = np.atleast_2d(np.array(grads))
	
	if X is not None and fX is not None: 
		X = np.array(X)
		fX = np.array(fX).flatten()
		assert X.shape[0] == fX.shape[0], "Dimensions of input do not match"

		opg_grads = perplexity_opg_grads(X, fX)
	else:
		opg_grads = np.zeros((0, grads.shape[1]))

	if grads is None:
		grads = np.zeros((0,X.shape[1]))

	all_grads = np.vstack([grads, opg_grads])
	U, s, VT = np.linalg.svd(all_grads.T, full_matrices = False)
	return U[:,0:subspace_dimension]

