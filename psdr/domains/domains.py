"""Base domain types"""
from __future__ import print_function, division

import types
import itertools
import numpy as np
from scipy.optimize import newton, brentq
from copy import deepcopy

import scipy.linalg
import scipy.stats
from scipy.optimize import nnls, minimize
from scipy.linalg import orth, solve_triangular
from scipy.spatial import ConvexHull
from scipy.stats import ortho_group
from scipy.spatial.distance import pdist
import cvxpy as cp

import sobol_seq

TOL = 1e-5

from ..quadrature import *
from ..misc import *
from ..exceptions import *

from .domain import *
from .euclidean import EuclideanDomain

__all__ = [
		'UnboundedDomain',
		'LinQuadDomain',
		'LinIneqDomain',
		'ConvexHullDomain',
		'BoxDomain', 
		'PointDomain',
		'UniformDomain',
		'RandomDomain',
		'NormalDomain',
		'LogNormalDomain',
		'TensorProductDomain',
	] 

# NOTE: These three functions need to be defined outside of the classes
# so we can call them on both LinQuadDomains and TensorProductDomains,
# the latter of which may not necessarily be a LinQuadDomain.

def closest_point(dom, x0, L, **kwargs):
	r""" Solve the closest point problem given a domain
	"""

	if dom.isinside(x0):
		return np.copy(x0)

	x_norm = cp.Variable(len(dom))
	constraints = dom._build_constraints_norm(x_norm)
	x0_norm =  dom.normalize(x0)
	
	if L is None:
		L = np.eye(len(dom))
		
	D = dom._unnormalize_der() 	
	LD = L.dot(D)
	obj = cp.norm(LD*x_norm - LD.dot(x0_norm))

	problem = cp.Problem(cp.Minimize(obj), constraints)
	problem.solve(**kwargs)

	# TODO: Check solution state 			
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))	

def constrained_least_squares(dom, A, b, **kwargs):
	x_norm = cp.Variable(len(dom))
	D = dom._unnormalize_der() 
	c = dom._center()	
		
	# \| A x - b\|_2 
	obj = cp.norm(x_norm.__rmatmul__(A.dot(D)) - b - A.dot(c) )
	constraints = dom._build_constraints_norm(x_norm)
	problem = cp.Problem(cp.Minimize(obj), constraints)
	problem.solve(**kwargs)
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))

def corner(dom, p, **kwargs):
	# If we already know the domain is empty, error early
	try:
		if dom._empty:
			raise EmptyDomainException
	except AttributeError:
		pass

	# Find the corner using CVXPY

	local_kwargs = merge(dom.kwargs, kwargs)		
	x_norm = cp.Variable(len(dom))
	D = dom._unnormalize_der() 	
		
	# p.T @ x
	if len(dom) > 1:
		obj = x_norm.__rmatmul__(D.dot(p).reshape(1,-1))
	else:
		obj = x_norm*float(D.dot(p))

	constraints = dom._build_constraints_norm(x_norm)
	problem = cp.Problem(cp.Maximize(obj), constraints)
	
	problem.solve(**local_kwargs)

	if problem.status in ['infeasible']:
		dom._empty = True
		raise EmptyDomainException	
	elif problem.status in ['unbounded']:
		dom._unbounded = True
		raise UnboundedDomainException
	elif problem.status not in ['optimal', 'optimal_inaccurate']:
		print(problem.status)
		raise SolverError

	# If we have found a solution, then the domain is not empty
	dom._empty = False
	return dom.unnormalize(np.array(x_norm.value).reshape(len(dom)))




		




	





