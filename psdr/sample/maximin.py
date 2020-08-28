from __future__ import print_function, division
import numpy as np
from scipy.spatial.distance import pdist
from iterprinter import IterationPrinter 

from ..geometry import voronoi_vertex_sample 

from .util import low_rank_L
from .initial import initial_sample


def maximin_block(domain, Nsamp, L = None, xtol = 1e-6, verbose = False, maxiter = 500, Xhat = None):
	r""" Construct a maximin design by block coordinate descent


	Given a domain :math:`\mathcal{D}\subset \mathbb{R}^m` and a matrix :math:`\mathbf{L}`,
	this function constructs an :math:`N` point maximin design solving 

	.. math:: 

		\max_{\mathbf{x}_1,\ldots, \mathbf{x}_N \in \mathcal{D}} 
		\min_{i\ne j} \|\mathbf{L} (\mathbf{x}_i - \mathbf{x}_j)\|_2.

	Here we use a block coordinate descent approach, treating each :math:`\mathbf{x}_i`
	in sequence, maximizing the minimum distance by moving it to its nearest Voronoi vertex.
	This process is then repeated until the maximum number of iterations is exceeded
	or the points :math:`\mathbf{x}_i` move less than a specified tolerance. 

	This block coordinate descent approach was previously described in [SHSV03]_.
	Here we exploit the fact that for a linear inequality constrained domain,
	we can find solution to each block optimization problem by invoking 
	:meth:`psdr.voroni_vertex`.

	Parameters
	----------
	domain: Domain	
		Space on which to build design
	Nsamp: int
		Number of samples to construct 
	L: array-like (*,m); optional
		Matrix defining the metric in which we seek to construct the maximin design.
		By default, this is the identity matrix.
	xtol: float, positive; optional
		Stopping criteria for movement of points
	verbose: bool; optional
		If True, print convergence information
	maxiter: int; optional 
		Maximum number of iterations of block coordinate descent


	References
	----------
	.. [SHSV03] Erwin Stinstra, Dick den Hertog, Peter Stehouwer, Arjen Vestjens.
		Constrained Maximin Designs for Computer Experiments.
		Technometrics 2003, 45:4, 340-346, DOI: 10.1198/004017003000000168

	"""
	if L is None:
		L = np.eye(len(domain))
	else:
		L = low_rank_L(L)
	
	if L.shape[0] == 1:
		# In the case of a 1-D Lipschitz matrix, we can exploit
		# the closed form solution -- namely uniformly spaced points between the corners
		c1 = domain.corner(L.flatten())
		c2 = domain.corner(-L.flatten())
		return np.vstack([(1-alpha)*c1 + c2*alpha for alpha in np.linspace(0,1, Nsamp)]) 	

	if Xhat is None:
		X = initial_sample(domain, L, Nsamp)
	else:
		X = Xhat

	if verbose:
		printer = IterationPrinter(it = '4d', maximin = '16.8e', dx = '10.3e')
		printer.print_header(it = 'iter', maximin = 'maximin dist.', dx = 'Δx')

	
	mask = np.ones(Nsamp, dtype = np.bool)
	for it in range(maxiter):
		max_move = 0
		for i in range(Nsamp):
			# Remove the current iterate 
			mask[i] = False
			Xt = X[mask,:]
			# Reset the mask
			mask[i] = True 
			x = voronoi_vertex_sample(domain, Xt, X[i], L = L, randomize = False)		
		
			# Compute movement of this point
			move = np.linalg.norm(L.dot(X[i] - x.flatten()), np.inf)
			max_move = max(max_move, move)
			
			# update this point
			X[i] = x
			
		if verbose:
			d = np.min(pdist( (L @ X.T).T))
			printer.print_iter(it = it, maximin = d, dx = max_move)

		# Only break at the end of a cycle
		if max_move < xtol:
			break
		
	return X
