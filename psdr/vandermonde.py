# Implementation of Vandermonde with Arnoldi ideas following BNT19x


import numpy as np
from scipy.special import comb

from .basis import index_set


def _update_vec(idx, ids):
	# Determine which column to multiply by
	diff = idx - ids
	# Here we pick the most recent column that is one off
	j = np.max(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
	i = int(np.argwhere(diff[j] == -1))
	return i, j	


def vandermonde_arnoldi(X, p):
	r"""


	Parameters
	----------
	X: array-like (M, n)
		Samples where the Vandermonde matrix is to be constructed

	p: int
		Polynomial degree

	Returns
	-------
	Q: np.array
		Matrix with orthogonal columns 
	H: np.array
		Upper Hessenberg matrix
	""" 

	X = np.array(X)
	M, n = X.shape
	p = int(p)
	
	idx = index_set(p, n)

	# Allocate memory for matrices
	Q = np.zeros((M, len(idx)))
	H = np.zeros((len(idx)+1, len(idx)))


	for k, ids in enumerate(idx):
		# Determine which column to multiply by
		if k == 0:
			q = np.ones(M)
		else:
			## Determine which column to multiply by
			#diff = idx[:k] - ids
			## Here we pick the most recent column that is one off
			#j = np.max(np.argwhere( (np.sum(np.abs(diff), axis = 1) <= 1) & (np.min(diff, axis = 1) == -1)))
			#i = int(np.argwhere(diff[j] == -1))
			#print(ids, j, idx[j], i) 
			i, j = _update_vec(idx[:k], ids)
			q = X[:,i] * Q[:,j]

		# Arnoldi
		for j in range(k):
			H[j,k] = Q[:,j].T @ q/M
			q -= H[j,k]*Q[:,j]

		H[k+1,k] = np.linalg.norm(q)/np.sqrt(M)
		Q[:,k] = q/H[k+1,k]
	return Q, H


def polyfitA(X, fX, p):
	r"""

	Parameters
	----------
	X: array-like (M,m)
		Sample locations
	fX: array-like (M,)
		Value of the function
	p: int
		Total degree of polynomial
	"""
	Q, H = vandermonde_arnoldi(X, p)
	d = np.linalg.lstsq(Q, fX, rcond = None)[0]
	return d, H

def polyvalA(d, H, X):
	r"""

	Parameters
	----------
	d: array-like (N,)
		Coefficients of polynomial from polyfitA
	H: array-like (N+1, N)
		Upper Hessenberg matrix from polyfitA
	X: array-like (M, m)
		New samples to evaluate the polynomial at
	"""
	X = np.atleast_2d(np.array(X))
	M, m = X.shape

	d = np.array(d)
	H = np.array(H)
	N = H.shape[1]
	
	# Determine polynomial degree
	p = 0
	while True:
		if comb(m + p, p) >= len(d):
			break
		else:
			p += 1
	idx = index_set(p, m)
	assert len(idx) == N
	
	W = np.zeros((M, N))

	for k, ids in enumerate(idx):
		if k == 0:
			w = np.ones(M)
		else:
			i, j = _update_vec(idx[:k], ids)
			w = X[:,i] * W[:,j]

		for j in range(k):
			w -= H[j,k] * W[:,j]
		W[:,k] = w/H[k+1,k]

	return W.dot(d)
