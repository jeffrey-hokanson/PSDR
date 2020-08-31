import numpy as np
import psdr, psdr.demos
from psdr.pgf import PGF
import seaborn as sns
import matplotlib.pyplot as plt

funs = [psdr.demos.OTLCircuit(), 
		psdr.demos.Borehole(),
		psdr.demos.WingWeight()]
names = ['OTLCircuit', 'Borehole', 'WingWeight']

algs = [lambda dom, M, L: psdr.random_sample(dom, M), 
		lambda dom, M, L: psdr.latin_hypercube_maximin(dom, M, maxiter = 1),
		lambda dom, M, L: psdr.minimax_lloyd(dom, M, L = L)	
		]
alg_names = ['random', 'LHS', 'minimax']
# Number of repetitions
Ms = [100,100, 10]
#Ms = [1,1,1]


Nsamp = 20

for fun, name in zip(funs, names):
	# Estimate the Lipschitz matrix
	np.random.seed(0)
	X = np.vstack([
		fun.domain.sample(1000),
		fun.domain.sample_grid(2)
		])
	grads = fun.grad(X)
	
	lip = psdr.LipschitzMatrix(verbose = True, reltol = 1e-7, abstol = 1e-7, feastol = 1e-7)
	lip.fit(grads = grads)

	L = lip.L
	
	# Samples to use when estimating dispersion
	#X0 = psdr.maximin_coffeehouse(fun.domain, 5000, L = L, N0 = 50)
	X0 = np.vstack([psdr.random_sample(fun.domain, 5000), fun.domain.sample_grid(2)])
	# Now perform designs
	for alg, alg_name, M in zip(algs, alg_names, Ms):
		dispersion = []
		# We can get a very sloppy fit with more points
		for i in range(M):
			np.random.seed(i)
			X = alg(fun.domain, Nsamp, L)
			dist = psdr.fill_distance_estimate(fun.domain, X, L = L, X0 = np.copy(X0)) 
			dispersion.append(dist)
			print(f'{alg_name:20s} : {i:4d} dispersion {dist:10.5e}')
	
		fig = plt.figure()
		ax = sns.swarmplot(dispersion)
		x, y = np.array(ax.collections[0].get_offsets()).T
		pgf = PGF()
		pgf.add('x', x)
		pgf.add('y', y)
		pgf.write(f'data/fig_design_{name}_{alg_name}.dat')	
		plt.clf()	
