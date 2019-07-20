from __future__ import division, print_function

import numpy as np
import scipy

def low_rank_L(L):
	r""" Convert a Lipschitz matrix into a low rank version

	Parameters
	----------
	L: array-like (*,m)
		Input matrix defining a Lipshitz-like metric

	Returns
	-------
	J: numpy.array (*,m)
		Version of L with full row rank
	"""
	L = np.atleast_2d(L)
	_, s, VT = scipy.linalg.svd(L)
	
	I = np.argwhere(~np.isclose(s,0)).flatten()
	if np.all(I):
		return L
	else:
		U = VT.T[:,I]
		# An explicit, low-rank version of L
		J = np.diag(s[I]).dot(U.T)
		return J
