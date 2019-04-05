from __future__ import print_function

import numpy as np
from scipy.spatial.distance import cdist
import scipy.linalg

from .domains import Domain

def voronoi_vertex(domain, Xhat, X0):
	r""" Constructs a subset of the Voronoi vertices on a given domain 


	Given a domain :math:`\mathcal{D} \subset \mathbb{R}^m` and a set of points
	:math:`\lbrace \widehat{\mathbf{x}}_j \rbrace_{j=1}^M`, the
	vertices of the bounded Voronoi diagram :math:`\mathbf{v}_i \in \mathcal{D}`
	are those points are those points that satisfy :math:`r` equidistance constraints

	.. math::

		\| \mathbf{v}_i - \widehat{\mathbf{x}}_{\mathcal{I}_i[1]}\|_2 =
		\| \mathbf{v}_i - \widehat{\mathbf{x}}_{\mathcal{I}_i[2]}\|_2 = \ldots = 
		\| \mathbf{v}_i - \widehat{\mathbf{x}}_{\mathcal{I}_i[r]}\|_2 

	and :math:`m-r` constraints from the domain.
	
	
	The bounded Voronoi vertices include the vertices of the domain.
	The number of domain vertices can grow exponentially in dimension (e.g., consider the cube),
	but the asymptotic cost of finding these is :math:`\mathcal{O}(mn)` [wikiVEP]_ 
	per vertex where :math:`n` is the number of linear inequality constraints specifying the domain.
	Hence, the complexity of finding the Voronoi vertices grows exponentially in the 
	dimension of the domain.
	Thus we use an approach due Lindemann and Cheng [LC05]_ to find a subset of vertices
	by taking a starting point :math:`\mathbf{x}_0` and sequentially projecting it
	onto a series of hyperplanes such that after :math:`m` steps we satisfy :math:`m` constraints.	

	Parameters
	----------
	domain: Domain
		Domain on which to construct the vertices
	Xhat: array-like (M, m)
		M existing points on the domain which define the Voronoi diagram
	X0: array-like (N, m)
		Initial points to use to find vertices		

	Returns
	-------
	X: np.ndarray(N, m)
		Points satisfying m constraints


	References
	----------
	.. [LC05] Iteratively Locating Voronoi Vertices for Dispersion Estimation
		Stephen R. Lindemann and Peng Cheng
		Proceedings of the 2005 Interational Conference on Robotics and Automation
	.. [wikiVEP] Vertex enumeration problem, Wikipedia.
		https://en.wikipedia.org/wiki/Vertex_enumeration_problem
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

	# This algorithm terminates when each point has m active constraints as we can have no more improvement
	# hence we substract the number of equality constraints to 	
	for k in range(m - domain.A_eq.shape[0]):
		# Find the nearest neighbors
		# As we intend to do this in high dimensions, 
		# we don't use a tree-based distance approach
		# as these don't scale well
		D = cdist(X0, Xhat)
		I = np.argsort(D, axis = 1)
		
		# set the search direction to move away from the closest point
		h = X0 - Xhat[I[:,0]]
		# which inequality constraints on the domain are active
		active = np.isclose(A.dot(X0.T).T , b)
		
		for i in range(X0.shape[0]):
			# For each point, project onto the feasible directions
			nullspace = [domain.A_eq.T]

			# active inequality constraints
			for j in range(A.shape[0]):
				if active[i,j] : nullspace += [A[j].reshape(-1,1)]

			# constraints from search directions
			for j in range(1, k+1):
				if np.isclose(D[i,I[i,0]], D[i,I[i,j]]):
					nullspace += [ (Xhat[I[i,0]] - Xhat[I[i,j]]).reshape(-1,1)]
					h[i] += (X0[i] - Xhat[I[i,j]])

			nullspace = np.hstack(nullspace)
			# If there are no active constraints, don't do anything
			if nullspace.shape[1]>0:
				Q, R = scipy.linalg.qr(nullspace, overwrite_a = True, mode = 'economic')
				print('k=%d: %d' % (k, Q.shape[1]))
				h[i] -= Q.dot(Q.T.dot(h[i]))
		
		# Now we find the furthest we can step along this direction before either hitting a
		# (1) separating hyperplane separating x0 and points in Xhat or 
		# (2) the boundary of the domain

		alpha = np.inf*np.ones(X0.shape[0])

		# (1) Take a step until we run into a separating hyperplane
		for xhat in Xhat:
			# we setup a hyperplane (p - p0)^* n  = 0 
			# where p0 is a point on the hyperplane
			# separating the closest point Xhat[:,I[:,0]] and xhat
			p0 = 0.5*(xhat + Xhat[I[:,0]])
			n = xhat - Xhat[I[:,0]]

			# Inner product of normal n with search direction h
			nh = np.sum(n*h, axis = 1)

			# Inner product for the numerator (x0 - p0)^* n
			numerator = -np.sum( (X0 - p0)*n, axis = 1)
			
			alpha_c = np.inf*np.ones(alpha.shape)
			act = ~np.isclose(np.abs(nh), 0) 
			alpha_c[act] = numerator[act]/nh[act]
			# When xhat is the closest point, we get a divide by zero error
			# We cannot move backwards, so these are also set to infinity
			alpha_c[alpha_c < 0 ] = np.inf
			alpha = np.minimum(alpha, alpha_c)

		# (2) Find intersection with the domain
		# cf., Domain._extent_ineq 
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

	return X0
