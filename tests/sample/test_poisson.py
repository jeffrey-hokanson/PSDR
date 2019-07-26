import numpy as np
from scipy.spatial.distance import pdist, squareform
import psdr


def test_poisson_disk_sample(m = 2, r = 0.3):
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = psdr.poisson_disk_sample(dom, r)
	D = squareform(pdist(X))
	D += np.diag(np.nan*np.ones(D.shape[0]))
	d = np.nanmin(D, axis = 1)
	assert np.all(d >= r) and np.all(d <= 2*r)
	

	
