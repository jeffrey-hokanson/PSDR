import numpy as np
import psdr, psdr.demos

def test_subspace_convergence():
	mego = psdr.ActiveSubspace()
	Ms = np.logspace(0, 4, 10)
	fun = psdr.demos.Borehole()
	sampler = psdr.random_sample
	ang, Ms = psdr.subspace_convergence(mego, fun, sampler, Ms, data = 'grad', subspace_dimension = 1)

if __name__ == '__main__':
	test_subspace_convergence()
