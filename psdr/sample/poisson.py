from __future__ import print_function, division
import numpy as np
from scipy.spatial.distance import cdist

def poisson_disk_sample(domain, r, L = None, Ntries = 100):
	r""" Implements a Poisson disk sampling on an arbitrary domain


	Parameters
	----------
	domain: Domain
		Domain on which to construct the sampling 
	r: float, positive
		Minimum separation between points
	"""


	if L is None:
		L = np.eye(len(domain))

	X = [domain.sample()]
	active = [True]
	while True:
		# Pick a random active elemenit
		try:
			i = int(np.random.permutation(np.argwhere(active))[0])
		except IndexError:
			break
		success = False

		LX = L.dot(np.array(X).T).T
		for it in range(Ntries):
			# Generate a random direction
			p = domain.random_direction(X[i])
			p /= np.linalg.norm(L.dot(p))

			# TODO: Change distribution to sample annulus uniformly
			ri = np.random.uniform(r, 2*r)
	
			# Generate trial point
			xc = X[i] + ri*p

			if domain.isinside(xc):
				d = cdist(L.dot(xc).reshape(1,-1), LX)
				if np.min(d) > r:
					X.append(xc)
					active.append(True)
					success = True
					break

		if not success:
			active[i] = False

	return np.array(X)
