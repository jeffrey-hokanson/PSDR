from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist

from .domains import Domain

def voronoi_vertex(domain, Xhat, X0):
	r""" Constructs a subset of the Voronoi vertices on a given domain 



	Parameters
	----------
	domain: Domain
		Domain on which to construct the vertices
	Xhat: array-like (M, m)
		M existing points on the domain which define the Voronoi diagram
	X0: array-like (N, m)
		Initial points to use to find vertices		

	References
	----------
	.. [LC05] Iteratively Locating Voronoi Vertices for Dispersion Estimation
		Stephen R. Lindemann and Peng Cheng
		Proceedings of the 2005 Interational Conference on Robotics and Automation
	"""

	# Startup checks
	assert isinstance(domain, Domain), "domain must be an instance of the Domain class"
	assert len(domain.Ls) == 0, "Currently this does not support domains with quadratic constraints"
	Xhat = np.atleast_2d(np.array(Xhat))
	X0 = np.atleast_2d(np.array(X0))

	# Linear inequality constraints for the domain (including lower bound/upper bound box constraints)
	A = domain.A_aug
	b = domain.b_aug

	m = len(domain)

	for k in range(m):
		# Find the nearest neighbors
		# As we intend to do this in high dimensions, 
		# we don't use a tree-based distance approach
		# as these don't scale well
		D = cdist(X0, Xhat)
		I = np.argsort(D, axis = 1)
		
		# set the search direction to move away from the closest point
		h = X0 - Xhat[I[:,0]]
		
		# project search direction into null-space existing constraints
		for j in range(1, k+1):
			print(I[:,0])
			act = np.isclose(D[:,I[:,0]], D[:,I[:,j]])
			print(act)
			pass

		# project search direction into tangent cone based on the domain constraints 



		# Now we find the furthest we can step along this direction before either hitting a
		# (1) separating hyperplane separating x0 and points in Xhat or 
		# (2) the boundary of the domain

		alpha = np.inf*np.ones(X0.shape[0])

		# (1) Take a step until we run into a separating hyperplane
		for xhat in Xhat:
			# we setup a hyperplane (p - p0)^* n  = 0 
			# separating the closest point Xhat[:,I[:,0]] and xhat
			p0 = 0.5*(xhat + Xhat[I[:,0]])
			n = xhat - Xhat[I[:,0]]

			# Inner product of normal n with search direction h
			nh = np.sum(n*h, axis = 1)
			# Inner product for the numerator (x0 - p0)^* n
			numerator = -np.sum( (X0 - p0)*n, axis = 1)
			
			# When xhat is the closest point, we get a divide by zero error
			with np.errstate(divide = 'ignore', invalid = 'ignore'):
				alpha_c = numerator/nh
			# Ignore these
			alpha_c[~np.isfinite(alpha_c)] = np.inf
			# We cannot move backwards, so these are also set to infinity
			alpha_c[alpha_c <= 0 ] = np.inf
			alpha = np.minimum(alpha, alpha_c)

		# (2) Find intersection with the domain
		AX0 = A.dot(X0.T).T
		Ah = A.dot(h.T).T
		with np.errstate(divide = 'ignore', invalid = 'ignore'):
			alpha_c = (b - AX0)/Ah
		alpha_c[~np.isfinite(alpha_c)] = np.inf
		alpha_c[alpha_c <= 0] = np.inf
		alpha_c = np.min(alpha_c, axis = 1)
		alpha = np.minimum(alpha, alpha_c)

		# Now finally take the step
		X0 += alpha.reshape(-1,1)*h

		print(X0)
		break
