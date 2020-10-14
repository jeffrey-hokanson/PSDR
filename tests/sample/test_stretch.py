import numpy as np
from scipy.stats import kstest, uniform
import psdr
import pytest


@pytest.mark.parametrize("m", [5])
@pytest.mark.parametrize("ns", [
	[1], 
	[1,2], 
	[5,], 
	[5,5],
	[3,3],
	[3,4,5],
])
def test_stretch_sample(m, ns):
	np.random.seed(0)
	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = domain.sample(2)
	
	Ls = [np.random.randn(n, m) for n in ns]
	# Perform this process a few times
	for it in range(5):
		x = psdr.stretch_sample(domain, X, Ls)
		x_str = ''.join([f'{xi:8.4f}' for xi in x])
		print(x_str)
		X = np.vstack([X, x.reshape(1,-1)])
	assert np.all(domain.isinside(X)), "Not all points inside the domain"

	


@pytest.mark.parametrize("m", [5])
@pytest.mark.parametrize("n", [1,2,3])
def test_stretch_sample_uniform(m,n):
	r"""
	In the case of only one metric, this approach boils down to coffeehouse sampling
	and as such should be approximately uniform throughout the domain

	"""
	np.random.seed(0)	

	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = domain.sample(2)
	Ls = [np.random.randn(n,m) for i in range(1)]
	L = Ls[0]

	for it in range(50):
		x = psdr.stretch_sample(domain, X, Ls)
		x_str = ''.join([f'{xi:8.4f}' for xi in x])
		print(x_str)
		X = np.vstack([X, x.reshape(1,-1)])
	
	if L.shape[0] == 1:
		y = (L @ X.T).flatten()
		# Find the extent in this direction
		x1 = domain.corner(L.T.flatten())	
		x2 = domain.corner(-L.T.flatten())
		Lx1 = float(L @ x1)
		Lx2 = float(L @ x2)
		loc = min(Lx1, Lx2)
		scale = max(Lx1, Lx2) - loc
		stat, pvalue = kstest(y, 'uniform', args = (loc, scale))
		print(f"probability {pvalue:5.1e}")
		assert pvalue > 1e-3
	else:
		# as the Kolmogorov-Smirnov test in scipy only allows 1-d comparisions
		# we test uniformity by taking random directions along the linear combinations
		S = psdr.sample_sphere(n, 20)
		for s in S:
			sL = s @ L	
			x1 = domain.corner(sL.T.flatten())	
			x2 = domain.corner(-sL.T.flatten())
			Lx1 = float(sL @ x1)
			Lx2 = float(sL @ x2)
			loc = min(Lx1, Lx2)
			scale = max(Lx1, Lx2) - loc
			y = (sL @ X.T).flatten()
			stat, pvalue = kstest(y, 'uniform', args = (loc, scale))
			print(f"probability {pvalue:5.1e}")

			assert pvalue > 1e-3

if __name__ == '__main__':
	test_stretch_sample(5, [3,4, 5])
	#test_stretch_sample_uniform(5, 1)
	#test_stretch_sample_uniform(5, 2)
