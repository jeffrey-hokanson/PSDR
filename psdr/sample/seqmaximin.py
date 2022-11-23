from __future__ import division, print_function

import itertools
import numpy as np
import scipy
from scipy.spatial.distance import cdist

from ..geometry import voronoi_vertex_sample, unique_points
from .initial import initial_sample

def seq_maximin_sample(domain, Xhat, Ls = None, Nsamp = int(1e3), X0 = None, slack = 0.9):
	r""" A multi-objective sequential maximin sampling 
	

	Given an existing set of samples :math:`\lbrace \widehat{\mathbf{x}}_j\rbrace_{j=1}^M\subset \mathcal{D}`
	from the domain :math:`\mathcal{D} \subset \mathbb{R}^m`, this algorithm finds a point :math:`\mathbf{x} \in \mathcal{D}`
	that approximately maximizes the distance of the new point to all other points in multiple distance metrics 
	give by matrices :math:`\mathbf{L}_i \in \mathbb{R}^{m\times m}`

	.. math::

		\max_{\mathbf{x} \in \mathcal{D}} \left\lbrace 
			\min_{j=1,\ldots,M} \|\mathbf{L}_i (\mathbf{x} - \widehat{\mathbf{x}}_j)\|_2
			\right\rbrace_{i}

	This algorithm uses :meth:`psdr.voronoi_vertex_sample` to generate local maximizers of this problem
	for each metric and then tries to greedily satisfy the distance requirements for each metric.

	A typical use case will have Ls that are of size (1,m)	
	This greedy sequential approach for constructing a maximin design is 
	the Coffee-House Designs of Muller [Mul01]_. However, the approach of Muller
	allows for a generic nonlinear solve for each sample point.  Here though
	we restrict the domain to a polytope specified by linear inequalities
	so we can invoke :meth:`psdr.voronoi_vertex_sample` to solve each step. 

	Parameters
	----------
	domain: Domain
		The domain from which we will be sampling
	Xhat: array-like (M, m)
		Previously existing samples from the domain 
	Ls: list of array-like (?, m) matrices, optional
		The weight matrix (e.g., Lipschitz matrix) corresponding to each metric;
		defaults to the identity matrix
	Nsamp: int, optional (default 1000)
		Number of samples to use when finding Voronoi vertices
	slack: float [0,1], optional (default 0.1)
		Rather than taking the point that maximizes the product of the
		distances in each metric, we choose the point x with greatest unweighted Euclidean
		distance from those candidates that are at least slack times the score of the best.

	References
	----------
	.. [Mul01] Coffee-House Designs.
		Werner G. Muller
		in Optimimum Design 2000, A. Atkinson et al. eds., 2001
	"""
	if Ls is None:
		Ls = [np.eye(len(domain))]

	# If we have no samples we pick a corner in the direction
	# of the dominant singular vector of the stacked L matrices
	if len(Xhat) == 0:
		Lall = np.vstack(Ls)
		_, s, VT = scipy.linalg.svd(Lall)
		# If several singular values are close, we randomly select a direction 
		# from that subspace
		I = np.argwhere(np.isclose(s, s[0])).flatten()
		u = VT.T[:,I].dot(np.random.randn(len(I)))
		return domain.corner(u)
	
	Xhat = np.array(Xhat)
	Xhat = np.atleast_2d(Xhat)

	#############################################################################
	# Otherwise, we proceed with identifiying Voronoi vertices associated with
	# each of the metrics (L's) provided.
	#############################################################################

	vertices = []
	distances = []
	for k, L in enumerate(Ls):
		# Find initial samples well separated
		if X0 is None:
			X = initial_sample(domain, L, Nsamp = Nsamp//(len(Ls)+1))
		else:
			X = np.copy(X0)

		# find the Voronoi vertices; we don't randomize as we are only interested
		# in the component that satisfies the constraint
		vert = voronoi_vertex_sample(domain, Xhat, X, L = L, randomize = False) 
		
		# Remove duplicates in the L norm
		I = unique_points(L.dot(vert.T).T)
		vert = vert[I]

		# Compute the distances between points in this metric
		D = cdist(L.dot(vert.T).T, L.dot(Xhat.T).T)
		D = np.min(D, axis = 1)
		# Order the vertices in decreasing distance
		I = np.argsort(-D)
		vert = vert[I]
		vertices.append(vert)
		distances.append(D[I])
		
	#############################################################################
	# Now we construct a number of candidate domains to sample from.
	# Many of these may be empty because the constraints are collectively infeasible
	#############################################################################

	# When generating these domains, we limit the number of vertices we consider
	max_verts = max(2, int(np.floor(1e2**(1./len(Ls) ))))

	# A list of which vertices to consider at each step
	coords = []
	for dist, vert in zip(distances, vertices):
		#if dist[0] == np.max([d[0] for d in distances]):
		#	# If this coordinate has the largest distance, we only sample the largest one 
		#	coords.append([0])
		#else:
		# Otherwise we sample the first few largest
		coords.append(np.arange(min(len(vert),max_verts)))

	# Generate a score associated with each
	# This score is the product to the distances in each metric 
	idx = list(itertools.product(*coords))
	dist_prod = [ sum([np.log10(dist[i]) for dist, i in zip(distances, idx_i)]) for idx_i in idx]

	# Order these in decreasing score
	I = np.argsort(-np.array(dist_prod))
	idx = [idx[i] for i in I]

	Xcan = []
	score_Ls = []
	used_idx = []
	for it in range(100):
		new_domain = False
		while len(idx) > 0:
			# Grab a combination of constraints to try
			idx_i = idx.pop(0)

			# These are the used indices; negative meaning no constraint applied
			found_idx = -1*np.ones(len(idx_i))

			domain_samp = domain	
			# Add the constraints on iteratively in decreasing distance
			for k in np.argsort([-dist[i] for i, dist in zip(idx_i, distances)]):
				L = Ls[k]
				vert = vertices[k][idx_i[k]]
				domain_test = domain_samp.add_constraints(A_eq = L, b_eq = L.dot(vert) )
				if domain_test.is_empty:
					#print("empty after %d constraints" % k)
					break
				else:
					domain_samp = domain_test
					found_idx[k] = idx_i[k]

			#print('found_idx', found_idx)
			#if found_idx not in used_idx:
			if len(used_idx) == 0 or np.min([np.linalg.norm(found_idx - used_idx_i) for used_idx_i in used_idx]) > 0:
				used_idx.append(found_idx)
				new_domain = True
				break

		if not new_domain:
			break
		
		# Generate candidates 
		X0 = initial_sample(domain_samp, np.eye(len(domain)), Nsamp = 100)
		Xcan_new = voronoi_vertex_sample(domain_samp, Xhat, X0)
		
		# Score samples: product of distances in each of the L metrics
		score_Ls_new = np.ones(Xcan_new.shape[0])
		for L in Ls:
			D = cdist(L.dot(Xcan_new.T).T, L.dot(Xhat.T).T)
			d = np.min(D, axis = 1)
			with np.errstate(divide='ignore'):
				score_Ls_new *= d # np.log10(d)
		score_Ls = np.hstack([score_Ls, score_Ls_new])
		Xcan.append(Xcan_new)
		
		# If remaining candidates are too close, break
		# (and we've used all the constraints)
		#print("score", np.max(score_Ls_new), "best", np.max(score_Ls), "b_eq", domain_samp.b_eq, "idx_i", idx_i)
		active_constraints = np.sum(found_idx >=0)
		if np.max(score_Ls_new) < slack*np.max(score_Ls) and active_constraints == len(Ls):
			# This prevents us from generating candidates that will be removed 
			#print("stopping")
			break

	#print("done with sampling")
	
	Xcan = np.vstack(Xcan)
	# Remove duplicates
	I = unique_points(Xcan)
	Xcan = Xcan[I]	
	score_Ls = score_Ls[I]
	
	
	# Compute Euclidean distances	
	D = cdist(Xcan, Xhat)
	score_I = np.min(D, axis = 1)
	
#	for i in np.argsort(-score_Ls):
#		print("%3d\t %g \t %g" % (i, score_Ls[i], score_I[i]))

#	import matplotlib.pyplot as plt
#	fig, ax = plt.subplots()
#	ax.plot(score_Ls, score_I, 'k.')
#	ax.set_xlabel('Ls score')
#	ax.set_ylabel('I score')
#	plt.show()

	# Now select the one within 95% of optimum 
	I = (score_Ls >= np.max(score_Ls)*slack)
	# Delete those not matching critera
	Xcan = Xcan[I]
	score_I = score_I[I]
	# pick the remaining point with highest Euclidean metric
	i = np.argmax(score_I)
	return Xcan[i]
