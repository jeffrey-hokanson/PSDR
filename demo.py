import numpy as np
import psdr
import psdr.demos
from scipy.spatial.distance import pdist, squareform
fun = psdr.demos.OTLCircuit()

if False:
	X = fun.domain.sample_grid(2)
	X = np.vstack([X, fun.domain.sample(5)])
	L1 = np.ones((1, len(fun.domain)))
	L2 = np.zeros((1,len(fun.domain)))
	L2[0,3] = 1.
	Ls = [L1, L2]

	x = psdr.seq_maximin_sample(fun.domain, X, Ls = Ls)

if True:
	domain = fun.domain
	L1 = np.random.randn(1,len(domain))
	L2 = np.random.randn(1,len(domain))
	
	domain = psdr.BoxDomain([-1, -1, -1], [1, 1, 1])
	L1 = np.ones((1, len(domain)))
	L1 = None
	#psdr.lipschitz_sample(domain, 7, [L1,L2])
	#X = psdr.sample.minimax_sample(domain, 5, L = L1, verbose = True, maxiter = 200)
	X = psdr.maximin_sample(fun.domain, 20, L = L1, verbose = True)
	print(X)
	#print(X[:,:2])
	#print(pdist(X[:,:2]))
