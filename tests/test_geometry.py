import numpy as np
from scipy.spatial.distance import pdist
import psdr

def test_unique_points(M = 100, m = 5):
	X = np.random.randn(M, m)
	X = np.vstack([X, X])
	
	I = psdr.geometry.unique_points(X)
	X2 = X[I]
	d = pdist(X2)
	assert np.min(d) > 0, "Points not unique"


def test_sample_sphere():
	X = psdr.geometry.sample_sphere(10, 100)
	assert X.shape == (100, 10)
	assert np.all(np.isclose(np.sum(X**2, axis = 1), 1))

def test_sample_simplex(dim = 5, Nsamp = 10):
	X = psdr.geometry.sample_simplex(dim, Nsamp)

	assert X.shape == (Nsamp, dim)
	assert np.all(np.isclose(np.sum(X, axis = 1), 1))
	assert np.all(X>=0)
	assert np.all(X<=1.)
