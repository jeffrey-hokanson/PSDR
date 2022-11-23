import numpy as np
import cvxpy as cp
from iterprinter import IterationPrinter

from ..geometry import voronoi_vertex, voronoi_vertex_sample, unique_points, cdist
from .maximin_coffeehouse import maximin_coffeehouse
from .maximin import maximin_block
from .minimax import minimax_design_1d


def _update_voronoi_sample(domain, Xhat, X0, L, M0):
	r""" Use randomized sampling to identify a subset of the bounded Voronoi vertices


	Basic idea: recycle found Voronoi iterates from previous iteration
	and keep adding new ones until we have the target number of points M0

	Parameters
	----------
	domain: LinIneqDomain

	Xhat: np.array (M, len(domain))
		Nodes to compute Voronoi vertices of
	X0: np.array (*, len(domain))
		Initial samples 
	L: np.array (len(domain), len(domain))
		Lipschitz matrix
	M0: int
		Target number of queries
	"""

	# Update the estimate Voronoi vertices
	V = voronoi_vertex_sample(domain, Xhat, X0, L = L)
	
	# Remove duplicates
	V = V[unique_points(V)]

	# we need to ensure every node has at least one voronoi vertex associated with it
	D = cdist(Xhat, V, L =L)
	d = np.min(D, axis = 0)
	for k in range(len(Xhat)):
		I = np.isclose(D[k], d)
		if np.sum(I) < len(domain):
			# Generate perturbations of nearby point
			X0 = np.outer(np.ones(2*len(domain)), Xhat[k])  + 1e-7*np.random.randn(2*len(domain), len(domain))
			# This pushes these points to nearby bounded Voronoi vertices
			Vnew = voronoi_vertex_sample(domain, Xhat, X0, L = L)
			V = np.vstack([V, Vnew])

	# Remove duplicates
	return V[unique_points(V)]


def _update_voronoi_full(domain, Xhat, X0, L, M0):
	return voronoi_vertex(domain, Xhat, L = L)


def minimax_lloyd(domain, M, L = None, maxiter = 100, Xhat = None, verbose = True, xtol = 1e-5, full = None):
	r""" A fixed point iteration for a minimax design

	This algorithm can be interpreted as a block coordinate descent type algorithm
	for the optimal minimax experimental design on the given domain.

	
	SD96.	
	"""
	# Terminate early if we have a simple case
	try:
		return minimax_design_1d(domain, M, L = L)
	except AssertionError:
		pass

	if full is None:
		if len(domain) < 3:
			_update_voronoi = _update_voronoi_full
		else:
			_update_voronoi = _update_voronoi_sample
	elif full is True:
		_update_voronoi = _update_voronoi_full
	elif full is False:
		_update_voronoi = _update_voronoi_sample

	if L is None:
		L = np.eye(len(domain))
	
	M0 = 10*len(domain)*M
	X0 = domain.sample(M0)

	if Xhat is None:
		if verbose: print(10*'='+" Building Coffeehouse Design " + 10*'=')
		Xhat = maximin_coffeehouse(domain, M, L, verbose = verbose)
		if verbose: print('\n'+10*'='+" Building Maximin Design " + 10*'=')
		Xhat = maximin_block(domain, M, L = L, maxiter = 50, verbose =verbose, X0 = Xhat)
		# The Shrinkage suggested by Pro17 hasn't been demonstrated to be useful with this initialization, so we avoid it
		if verbose: print('\n'+10*'='+" Building Minimax Design " + 10*'=')

	x = cp.Variable(len(domain))
	c = cp.Variable(len(domain))
	constraints = domain._build_constraints(x)

	if verbose:
		printer = IterationPrinter(it = '4d', minimax = '18.10e', dx = '9.3e')
		printer.print_header(it = 'iter', minimax = 'minimax est', dx = 'Δx')



	V = domain.sample(M0)
	Xhat_new = np.zeros_like(Xhat)

	for it in range(maxiter):

		# Compute new Voronoi vertices
		V = _update_voronoi(domain, Xhat, V, L, M0)	
		D = cdist(Xhat, V, L = L)
		d = np.min(D, axis = 0)
	
		for k in range(M):
			# Identify closest points to Xhat[k]
			I = np.isclose(D[k], d)
		
			# Move the Xhat[k] to the circumcenter 	
			ones = np.ones((1, np.sum(I)))
			obj = cp.mixed_norm( (L @ ( cp.reshape(x,(len(domain),1)) @ ones - V[I].T)).T, 2, np.inf)
			prob = cp.Problem(cp.Minimize(obj), constraints)
			prob.solve()
			Xhat_new[k] = x.value

		
		dx = np.max(np.sqrt(np.sum((Xhat_new - Xhat)**2, axis = 1)))

		if verbose:
			printer.print_iter(it = it, minimax = np.max(d), dx = dx) 
		
		Xhat[:,:] = Xhat_new[:,:]

		if dx < xtol:
			if verbose:
				print('small change in design')
			break

	return Xhat
