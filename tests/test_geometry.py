import numpy as np
from psdr.geometry import unique_points
from scipy.spatial.distance import pdist

def test_unique_points(M = 100, m = 5):
	X = np.random.randn(M, m)
	X = np.vstack([X, X])
	
	I = unique_points(X)
	X2 = X[I]
	d = pdist(X2)
	assert np.min(d) > 0, "Points not unique"
