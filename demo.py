import numpy as np
import psdr
import psdr.demos
from scipy.spatial.distance import pdist, squareform

if True:
	m = 3
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	X = dom.sample(500)
	
	r = 0.5
	psdr.minimax_covering_discrete(X, 0.5)
	


if False:
	fun = psdr.demos.OTLCircuit()
	X = fun.domain.sample_grid(2)
	X = np.vstack([X, fun.domain.sample(5)])
	L1 = np.ones((1, len(fun.domain)))
	L2 = np.zeros((1,len(fun.domain)))
	L2[0,3] = 1.
	Ls = [L1, L2]

	x = psdr.seq_maximin_sample(fun.domain, X, Ls = Ls)

if False:
	#domain = fun.domain
	domain = psdr.BoxDomain([-1, -1, -1], [1, 1, 1])
	L1 = np.random.randn(2,len(domain))
	L2 = np.random.randn(1,len(domain))

	I = np.eye(len(domain))
	#psdr.minimax_cluster(domain, 10, L = I, N0 = 100)
	X = psdr.poisson_disk_sample(domain, 0.5)
	print(X)
	D = squareform(pdist(X))
	D += np.diag(np.nan*np.ones(D.shape[0]))
	print(np.nanmin(D, axis = 1))	
	#L1 = np.ones((1, len(domain)))
	#L1 = None
	#X = psdr.lipschitz_sample(domain, 7, [L1,L2], verbose =True)
	#X = psdr.sample.minimax_sample(domain, 5, L = L1, verbose = True, maxiter = 200)
	#X = psdr.maximin_sample(fun.domain, 20, L = L1, verbose = True)
	#print(X)
	#print(X[:,:2])
	#print(pdist(X[:,:2]))
