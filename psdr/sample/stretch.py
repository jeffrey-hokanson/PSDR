r""" Stretched sampling routines
"""
import numpy as np
from .initial import initial_sample
from .maximin_coffeehouse import maximin_coffeehouse 
from ..geometry import voronoi_vertex_sample, cdist
from cvxpy.error import SolverError

def _maximin_block_candidate(domain, Xf, L):
	r"""
	"""
	X0 = initial_sample(domain, L, Nsamp = 2*len(Xf))
	#X0 = maximin_coffeehouse(domain, 2*len(Xf), L = L, N0 = 1, full = False, verbose = False)
	#X0 = domain.sample(2*len(Xf))
	X = voronoi_vertex_sample(domain, Xf, X0, L = L, randomize = False) 
	# find the point furthest away 
	D = cdist(Xf, X, L)
	d = np.min(D, axis = 0)
	k = np.argmax(d)
	return X[k], d[k]



def stretch_sample_domain(domain, X, Ls, verbose = False):
	r"""


	Parameters
	----------
	domain: Domain
		domain of dimension m on which to sample
	X: array-like (M, m)
		Existing samples on the domain
	Ls: list of arrays of size (*, m)
		List of metrics in which to perform dimension reduction	
	"""
	# Because we'll be popping items out of this list, we create a copy
	Ls = Ls.copy()
	# A copy of the domain we'll keep adding constraints to	
	domain = domain.copy()

	it = 0	
	while len(Ls) > 0:
		# Compute candidate points and distances
		xd = [list(_maximin_block_candidate(domain, X, L)) + [L, k] for k, L in enumerate(Ls)]

		# Find the largest distance
		x, d, L, k = max(xd, key = lambda l:l[1])
		# Remove this L from the list
		Ls.pop(k)


		# Try to add the new constraint
		domain_new = domain.add_constraints(A_eq = L, b_eq = L @ x)

	
		# If we have an empty domain, stop and return the existing domain
		# otherwise continue
		try:
			if domain_new.is_empty: 	break
			else: domain = domain_new
		except SolverError:
			break

		it += 1
		if verbose:
			print(f'iter {it:2d}, domain size {len(domain)}, eq constraints {len(domain.A_eq)}')

		# If we have more constraints than the dimension of the space, stop
		if len(domain.A_eq) > len(domain): break

	return domain

def stretch_sample(domain, Xf, Ls, verbose = False):
	domain_stretch = stretch_sample_domain(domain, Xf, Ls, verbose = verbose)
	#print(domain_stretch.A_eq)
	#print(domain_stretch.b_eq)
	X0 = domain_stretch.sample(100)
	X = voronoi_vertex_sample(domain_stretch, Xf, X0)
	D = cdist(Xf, X)
	d = np.min(D, axis = 0)
	k = np.argmax(d)
	return X[k]
	
 

class StretchSample:
	def __init__(self, fun, X = None, fX = None):
		self.fun = fun
		 	
