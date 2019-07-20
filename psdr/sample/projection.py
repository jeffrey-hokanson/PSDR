from __future__ import print_function

import numpy as np
from satyrn import picosat

from .util import low_rank_L

def projection_sample(domain, Nsamp, Ls, maxiter = 1000, verbose = False, _lhs = False):
	r""" Construct a maximin design with respect to multiple projections

	


	Parameters
	----------
	domain: Domain
		domain from which the samples are taken
	Nsamp: int
		Number of points to take
	Ls: list of arrays of size (*,m)
		Weighte metrics
	maxiter: int, optional
		Number of feasible designs that are randomly selected from
		the space of available solutions.
	verbose: bool, optional
		If True, print iteration information
	"""	 

	# If passed with _lhs, this algorithm constructs a generalized latin hypercube design
	if _lhs:
		I = np.eye(len(domain))
		Ls = [I[i].reshape(1,-1) for i in range(len(domain))]


	# Make sure none of the L matrices has low row rank
	Ls = [low_rank_L(L) for L in Ls]
	Lall = np.vstack(Ls)
	if Lall.shape[0] > len(domain):
		raise ValueError("Total dimension of Ls must not exceed the dimension of the domain")
	
	# Construct maximin points for each metric L
	ys = []
	As = [ [None for j in range(Nsamp)] for L in Ls]
	bs = [ [None for j in range(Nsamp)] for L in Ls]
	for i, L in enumerate(Ls):

		if _lhs:
			# If constructing an LHS design, we use minimax points in each projection 
			c1 = domain.corner(L.flatten())
			c2 = domain.corner(-L.flatten())
			xi = np.linspace(domain.norm_lb[i], domain.norm_ub[i], Nsamp + 1)
			X = (xi[1:]+xi[0:-1])/2.
		else:
			# We use maximin points in general as these generalize two and higher dimensional projections
			X = maximin_sample(domain, Nsamp, L)
		y = L.dot(X.T).T
		ys.append(y)

		# Construct the set of neighbors for each point
		if y.shape[1] == 1:
			# Since Qhull doesn't support 1-d, we explicitly compute the neighbors
			# for the 1-d case.
			I = np.argsort(y.flatten())
			neighbors = [ None for j in range(Nsamp)]
			for j in range(Nsamp):
				if j == 0:
					neighbors[I[j]] = np.array([I[j+1]], dtype = np.int)
				elif j == Nsamp - 1:
					neighbors[I[j]] = np.array([I[j-1]], dtype = np.int)
				else:
					neighbors[I[j]] = np.array([ I[j-1], I[j+1] ], dtype = np.int)
			
		else:
			delaunay = Delaunay(y)
			indices, indptr = delaunay.vertex_neighbor_vertices
			# Convert the Delaunay data structure into a list of neighbors
			# https://stackoverflow.com/a/23700182
			neighbors = [indptr[indices[k]:indices[k+1]] for k in range(len(y))]
		
		# Now construct the inequality constraints for each point's Voronoi cell
		for j, neigh in enumerate(neighbors):
			A = []
			b = []
			for k in neigh:
				p = 0.5*(X[j] + X[k]) 	# point on separating hyperplane
				n = (X[k] - X[j]).reshape(-1,1) 		# normal vector to hyperplane
				A.append( n.T.dot(L.T).dot(L))
				b.append( n.T.dot(L.T).dot(L).dot(p) )
			
			As[i][j] = np.vstack(A)
			bs[i][j] = np.hstack(b)
			dom = domain.add_constraints(A = As[i][j], b = bs[i][j])
			assert dom.isinside(X[j])
			assert np.sum(dom.isinside(X)) == 1	


	def encode(metric, order, value):
		# Python3 needs this explicitly an int to interface with pylgl
		return int(metric*Nsamp*Nsamp + order*Nsamp + value + 1)
	def decode(idx):
		return (idx-1) // Nsamp**2, ((idx-1) % Nsamp**2)// Nsamp, (idx-1) % Nsamp

	@lru_cache(maxsize = int(1e6) )
	def subdomain(idx):
		A = np.vstack([As[metric][idx[metric]] for metric in range(len(Ls))])
		b = np.hstack([bs[metric][idx[metric]] for metric in range(len(Ls))])
		return domain.add_constraints(A = A, b = b)
	
	@lru_cache(maxsize = int(1e6))
	def subdomain_sample(idx):
		# The point in the domain that is nearest to the target position
		subdom = subdomain(idx)
		A = np.vstack([Ls])
		b = np.hstack([y[i] for y, i in zip(ys, idx)])
		return subdom.constrained_least_squares(A, b)
			

	# Construct the geometric constraints we enforce with the SAT solver
	# i.e., no repetition	
	geo_cnf = []
	for metric in range(len(Ls)):
		for order in range(Nsamp):
			# At least one must be one
			geo_cnf += [ [encode(metric, order, value) for value in range(Nsamp)] ]	
			# Not more than one is on
			# we do this by checking not ( v1 and v2) which is equivalent to 
			# not v1 or not v2 by DeMorgan
			geo_cnf += [ [-encode(metric, order, v1), -encode(metric, order, v2)] 
						for v1, v2 in zip(*np.triu_indices(Nsamp,1)) ]

		# Each equality constraint cannot be selected more than once
		for value in range(Nsamp):
			geo_cnf += [ [-encode(metric, o1, value), -encode(metric, o2, value)] 
						for o1, o2 in zip(*np.triu_indices(Nsamp, 1)) ]

	# To fix ordering of the samples, we fix the ordering of the first metric
	geo_cnf += [ [encode(0, value, value)]  for value in range(Nsamp)]


	score_best = tuple([0. for i in range(Nsamp)])
	perms_best = []
	X_best = None

	# Initialize the solver and tie its random seed to numpy's random state
	sat_iter = picosat.itersolve(geo_cnf, initialization = 'random', seed = np.random.randint(2**16))

	it = 0
	while True:
		try:
			sol = sat_iter.next()
		except (KeyboardInterrupt, SystemExit):
			raise
		except StopIteration:
			try:
				# This automatically releases previous assumptions
				if verbose: print("assumptions failed")
				sol = sat_iter.next()
			except (KeyboardInterrupt, SystemExit):
				raise
			except:
				break
	
		# Convert the solution format into a permutation matrix
		sol = np.array(sol)
		perms = -np.ones((len(Ls), Nsamp), dtype = np.int)
		for i in np.argwhere(sol > 0).flatten():
			metric, order, value = decode(sol[i])
			perms[metric, order] = value

		# Construct a series of domains from this set of constraints
		subdoms = []
		for order in range(Nsamp):
			subdoms.append(subdomain(tuple(perms[:,order])))
	
		# Find domains that are empty and remove that combination
		# This happens here so we don't need to iterate over all combinations
		# at the start---something that is expensive in cases where only a small 
		# fraction of subdomains are invalid
		new_geo_cnf = []
		for order, subdom in enumerate(subdoms):
			if subdom.is_empty:
				# If the subdomain is empty, block future samples from using this one
				new_geo_cnf += [ [-encode(metric, o, perms[metric, order]) for metric in range(len(Ls))] 
						for o in range(Nsamp)] 

		sat_iter.add_clauses(new_geo_cnf)
		geo_cnf += new_geo_cnf

		if all([ not subdom.is_empty for subdom in subdoms]):
			print(perms[1:,:])
			it += 1	

	
			X = np.vstack([subdomain_sample(tuple(perms[:,order])) for order in range(Nsamp)])
			
			if Lall.shape[0] < len(domain):
				# If the Ls do not completely fix each of the samples, iterate to find maximin design
				consubdoms = [subdom.add_constraints(A_eq = Lall, b_eq = Lall.dot(x)) for subdom, x in zip(subdoms, X)]
				X = np.vstack([consubdom.sample() for consubdom in consubdoms])
				mask = np.ones(Nsamp, dtype = np.bool)
				for it2 in range(100):
					max_move = 0
					for i in range(Nsamp):
						# Remove the current iterate 
						mask[i] = False
						Xt = X[mask,:]
						# Reset the mask
						mask[i] = True 
						x = voronoi_vertex(consubdoms[i], Xt, X[i])	
	
						# Compute movement of this point
						move = np.linalg.norm(X[i] - x.flatten(), np.inf)
						max_move = max(max_move, move)
						
						# update this point
						X[i] = x
					if max_move < 1e-6:
						break
			
			D = squareform(pdist(X))
			D += np.max(D)*np.eye(D.shape[0])
			min_dist = np.min(D, axis = 0)
			score = tuple(np.sort(min_dist))
			
			if score > score_best:
				X_best = X
				score_best = score
				perms_best = perms
				updated = True
			
				# Force solver to keep best
				keep = np.argsort(-min_dist)[0:-len(Ls)*2]
				assumptions = []
				for k in keep:
					assumptions += [encode(metric, k, perms[metric,k]) for metric in range(len(Ls))]
			else: 
				updated = False

			sat_iter.assume(assumptions)
			
			if verbose:
				mess = "it %4d:" % (it,) 
				mess += 'best: '
				mess += ' '.join(['%3.2f' % s for s in score_best])
				if updated:
					mess += ' *updated*'
				print(mess)
		else:
			if verbose:
				print('found invalid domain')
		if it >= maxiter:
			break
	return X_best 

