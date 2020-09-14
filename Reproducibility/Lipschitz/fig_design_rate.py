import numpy as np
import psdr, psdr.demos
from psdr.pgf import PGF

from joblib import Memory 

memory = Memory('.cache',verbose = 0)
 
@memory.cache
def build_random_design(domain, M, L, seed):
	np.random.seed(seed)
	return psdr.random_sample(domain, M)

@memory.cache
def build_lhs_design(domain, M, L, seed):
	np.random.seed(seed)
	return psdr.latin_hypercube_maximin(domain, M, maxiter = 1)

@memory.cache
def build_minimax_design(domain, M, L, seed):
	np.random.seed(seed)
	return psdr.minimax_lloyd(domain, M, L = L)


@memory.cache
def design_dispersion(X, domain, L):
	X0 = domain.sample_grid(6)
	dist = psdr.fill_distance_estimate(domain, X, L = L, X0 = X0) 
	return dist

Ntrials = 1

Npoints = np.arange(5, 105, 5)

fun = psdr.demos.OTLCircuit()

# Estimate the Lipschitz matrix
lip = psdr.LipschitzMatrix(verbose = True, abstol = 1e-7, reltol = 1e-7, feastol = 1e-7)
lip.fit(grads = fun.grad(fun.domain.sample_grid(4)))
L = lip.L

algs = [build_random_design, build_lhs_design, build_minimax_design]
alg_names = ['random', 'lhs', 'minimax']

for alg, name in zip(algs, alg_names):
	p0 = np.zeros(len(Npoints))
	p25 = np.zeros(len(Npoints))
	p50 = np.zeros(len(Npoints))
	p75 = np.zeros(len(Npoints))
	p100 = np.zeros(len(Npoints))
	for k, N in enumerate(Npoints):
		dists = []
		for trial in range(Ntrials):
			X = alg(fun.domain, N, L, trial)
			dist = design_dispersion(X, fun.domain, L)
			dists.append(dist)
			print(f'M : {N:3d} | dispersion {dist:10.4e}')
		
		p0[k], p25[k], p50[k], p75[k], p100[k] = np.percentile(dists, [0, 25, 50, 75,100])

		# Write incremental data
		pgf = PGF()
		pgf.add('N', Npoints)
		pgf.add('p0', p0)
		pgf.add('p25', p25)
		pgf.add('p50', p50)
		pgf.add('p75', p75)
		pgf.add('p100', p100)
		
		pgf.write(f'data/fig_design_rate_{name}.dat')
	
 
