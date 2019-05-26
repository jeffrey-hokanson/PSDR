""" Test the Latin Hypercube designs """
import numpy as np
from scipy.spatial.distance import pdist
from itertools import product
import psdr


def test_lhs(m = 2):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))

	N = 10
	np.random.seed(0)
	for metric, jiggle in product(['maximin', 'corr'], [True, False]):
		
		X = dom.latin_hypercube(N, metric = metric, maxiter = 1000, jiggle = jiggle)
		
		# Check metric
		X0 = dom.latin_hypercube(N, metric = metric, maxiter = 1, jiggle = jiggle)
		if metric == 'maximin':
			assert np.min(pdist(X)) >= np.min(pdist(X0))

		if metric == 'corr':
			assert np.linalg.norm(np.eye(m) - np.corrcoef(X.T), np.inf) < \
				np.linalg.norm(np.eye(m) - np.corrcoef(X0.T), np.inf) 
	
		# Check spacing is uniform
		if jiggle is False:
			for i in range(m):
				x = np.sort(X[:,i])
				assert np.all(np.isclose(x[1:]-x[:-1], x[1] - x[0]))
						
