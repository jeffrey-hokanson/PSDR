import numpy as np
from iterprinter import IterationPrinter
from functools import lru_cache
from ..geometry import voronoi_vertex, voronoi_vertex_sample, unique_points, cdist
from scipy.optimize import minimize_scalar


def minimax_scale(domain, Xhat, L = None, full = None, npoints = 20, N0 = None, X0 = None, verbose = True):
	r""" Only scale points towards the center as 


	Parameters
	----------
	X0
		Starting points for voronoi_vertex_sample
	N0
		Number of starting points used if not given
	"""
	M = len(Xhat)

	if verbose:
		printer = IterationPrinter(scale = '9.5f', score = '16.8e')
		printer.print_header(scale = 'scale', score = 'minimax dist.')

	if N0 is None:
		N0 = 10*len(Xhat)
	
	if X0 is None:
		X0 = domain.sample(N0)

	if full is None:
		if len(domain) <= 3:
			full = True
		else:
			full = False
		

	def scale_design(alpha):
		return alpha*(Xhat - np.outer(np.ones(M), domain.center)) + domain.center

	@lru_cache()	
	def score_design_full(alpha):
		Xtmp = scale_design(alpha)
		V = voronoi_vertex(domain, Xhat, L = L)
		D = cdist(Xtmp, V, L = L)
		d = np.max(np.min(D, axis = 0))
		if verbose:
			printer.print_iter(scale = alpha, score = d)
		return d

	@lru_cache()
	def score_design_sample(alpha):
		Xtmp = scale_design(alpha)
		V = voronoi_vertex_sample(domain, Xhat, X0, L = L)
		D = cdist(Xtmp, V, L = L)
		d = np.max(np.min(D, axis = 0))
		if verbose:
			printer.print_iter(scale = alpha, score = d)
		return d

	if full:
		score_design = score_design_full
	else:
		score_design = score_design_sample

	alpha_vec = np.linspace(0.5, 1, npoints)
	score_vec = [score_design(alpha) for alpha in alpha_vec]
	
	k = np.argmin(score_vec)

	if verbose:
		printer.print_iter()
	try:
		res = minimize_scalar(score_design, (alpha_vec[k-1], alpha_vec[k], alpha_vec[k+1]), bounds =[0,1], options = {'maxiter': 10})
		alpha = res.x
	except (ValueError, IndexError) :
		alpha = alpha_vec[k]

	return scale_design(alpha)
