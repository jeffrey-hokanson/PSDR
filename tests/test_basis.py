import numpy as np
import scipy.linalg
from psdr import (MonomialTensorBasis, 
	LegendreTensorBasis, 
	ChebyshevTensorBasis, 
	LaguerreTensorBasis, 
	HermiteTensorBasis)



def test_equivalence(m = 3, p = 5):
	""" Check that these bases all express the same thing 
	by checking the range of their Vandermonde matrices coincide
	"""
	M = 100
	X = np.random.randn(M, m)
	bases = [MonomialTensorBasis(m, p),
		LegendreTensorBasis(m, p),
		ChebyshevTensorBasis(m, p),
		LaguerreTensorBasis(m, p),
		HermiteTensorBasis(m, p),
		]

	Vs = [basis.V(X) for  basis in bases]
	Us = [scipy.linalg.orth(V) for V in Vs]
	
	for U1 in Us:
		for U2 in Us:
			print scipy.linalg.svdvals(U1.T.dot(U2))
			assert np.all(np.isclose(scipy.linalg.svdvals(U1.T.dot(U2)), 1.))

		



