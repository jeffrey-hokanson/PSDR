import numpy as np
from ..geometry import voronoi_vertex, cdist
from iterprinter import IterationPrinter


def maximin_coffeehouse(domain, N, L = None, verbose = True, N0 = None, full = None):
	r""" Construct an N point maximin design that is 50% efficient


	This appears in [Pro17]_ in section 4.1.
	

	Parameters
	----------
	domain: Domain
		Domain on which to construct the design
	N: int
		Number of points to generate
	L: array-like (*,m)
		Lipschitz matrix defining a distance metric
	N0: int
		Number of samples to take when constructing voronoi vertices
		Defaults to 10*N
	full: [None, True, False]
		If True, use compute all the Voronoi vertices, otherwise compute a subsample.	
		If None, this choice is made based on the dimension of the domain, where if len(domain)<=3, 
		all the Voronoi vertices are computed.

	References
	----------
	[Mul01]_ Werner G. Muller
		Coffee-house designs.
		Appearing in "Optimum Design 2000", pp 241--248. Atkinson et al. eds.
	"""

	if full is None:
		if len(domain) <= 3:
			full = True
		else:
			full = False

	if L is None:
		L = np.eye(len(domain))
	if N0 is None:
		N0 = 10*N

	Xhat = [domain.sample()]
	X0 = domain.sample(N0)

	if verbose:
		printer = IterationPrinter(it = '4d', dist = '10.2e')
		printer.print_header(it = 'iter', dist = 'distance')

	for j in range(1,N):
		if full:
			X = voronoi_vertex(domain, np.array(Xhat), L = L) 
		else:
			X = voronoi_vertex_sample(domain, np.array(Xhat), X0, L = L)
		D = cdist(np.array(Xhat), X, L = L)
		d = np.min(D, axis = 0)
		k = np.argmax(d)
		Xhat.append(X[k])

		if verbose:
			printer.print_iter(it = j, dist = d[k])	

	return np.array(Xhat)	
