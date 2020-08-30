import numpy as np
import psdr 
from psdr.pgf import PGF
import matplotlib.pyplot as plt
import seaborn as sns

from joblib import Parallel, delayed
from joblib import Memory
memory = Memory('.cache', verbose = False)


m = 2			# Dimension
domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
M = 15  		# number of points in the design
L = np.eye(m)	# Lipschitz matrix


@memory.cache
def score_design(Xhat):
	V = psdr.voronoi_vertex(domain, Xhat, L)
	D = psdr.cdist(Xhat, V, L)
	return np.max(np.min(D, axis= 0))

@memory.cache
def random_design(seed):
	np.random.seed(seed)
	return domain.sample(M)

@memory.cache
def random_minimax_design(seed):
	Xhat = random_design(seed)
	Xhat = psdr.minimax_lloyd(domain, M, L = L, maxiter = 100, full = True, Xhat = Xhat, verbose = True)
	return Xhat

@memory.cache
def coffeehouse_design(seed):
	np.random.seed(seed)
	return psdr.maximin_coffeehouse(domain, M, L = L)	

@memory.cache
def coffeehouse_maximin_design(seed):
	Xhat = coffeehouse_design(seed)
	return psdr.maximin_block(domain, M, L = L, Xhat = Xhat, maxiter = 100, verbose = True) 

@memory.cache
def random_maximin_design(seed):
	Xhat = random_design(seed)
	return psdr.maximin_block(domain, M, L = L, Xhat = Xhat, maxiter = 100, verbose = True) 

@memory.cache
def coffeehouse_minimax_design(seed):
	Xhat = coffeehouse_design(seed)
	Xhat = psdr.minimax_lloyd(domain, M, L = L, maxiter = 100, full = True, Xhat = Xhat, verbose = True)
	return Xhat

@memory.cache
def coffeehouse_maximin_minimax_design(seed):
	Xhat = coffeehouse_maximin_design(seed)
	Xhat = psdr.minimax_lloyd(domain, M, L = L, maxiter = 100, full = True, Xhat = Xhat, verbose = True)
	return Xhat

@memory.cache
def random_maximin_minimax_design(seed):
	Xhat = random_maximin_design(seed)
	Xhat = psdr.minimax_lloyd(domain, M, L = L, maxiter = 100, full = True, Xhat = Xhat, verbose = True)
	return Xhat


@memory.cache
def coffeehouse_maximin_scale_minimax_design(seed):
	Xhat = coffeehouse_maximin_design(seed)
	Xhat = psdr.minimax_scale(domain, Xhat, L = L)
	Xhat = psdr.minimax_lloyd(domain, M, L = L, maxiter = 100, full = True, Xhat = Xhat, verbose = True)
	return Xhat

@memory.cache
def coffeehouse_scale_minimax_design(seed):
	Xhat = coffeehouse_design(seed)
	Xhat = psdr.minimax_scale(domain, Xhat, L = L)
	Xhat = psdr.minimax_lloyd(domain, M, L = L, maxiter = 100, full = True, Xhat = Xhat, verbose = True)
	return Xhat


# Actually run some code
seeds = np.arange(100)

random_score = [score_design(random_design(seed)) for seed in seeds]
coffeehouse_score = [score_design(coffeehouse_design(seed)) for seed in seeds]
random_minimax_score = [score_design(random_minimax_design(seed)) for seed in seeds]
coffeehouse_minimax_score = [score_design(coffeehouse_minimax_design(seed)) for seed in seeds]
coffeehouse_maximin_minimax_score = [score_design(coffeehouse_maximin_minimax_design(seed)) for seed in seeds]
coffeehouse_maximin_scale_minimax_score = [score_design(coffeehouse_maximin_scale_minimax_design(seed)) for seed in seeds]
random_maximin_minimax_score = [score_design(random_maximin_minimax_design(seed)) for seed in seeds]

# 
data = [
		random_minimax_score,
		random_maximin_minimax_score,
		coffeehouse_minimax_score, 
		coffeehouse_maximin_minimax_score,
		coffeehouse_maximin_scale_minimax_score,
	]

sns.set_style('whitegrid')
#ax = sns.violinplot(data = data, inner = None)
ax = sns.swarmplot(data = data)
ax.set_xticklabels([
	'random \n→ minimax', 
	'random \n→ maximin \n→ minimax', 
	'coffeehouse \n→ minimax', 
	'coffeehouse \n→ maximin \n→ minimax', 
	'coffeehouse \n→ maximin \n→ scale \n→ minimax',])

names = ['random_minimax',
	'random_maximin_minimax', 
	'coffeehouse_minimax',
	'coffeehouse_maximin_minimax',
	'coffeehouse_maximin_scale_minimax',
	]
	
for coll, name in zip(ax.collections, names):
	x, y = np.array(coll.get_offsets()).T
	print(x)
	print(y)
	pgf = PGF()
	pgf.add('x', x)
	pgf.add('y', y)
	pgf.write('data/fig_initialization_%s.dat' % name)

best = np.min(np.hstack(data))
print(best)
ax.axhline(best, color = 'k')

ax.set_xlabel('method')
ax.set_ylabel('minimax distance')
plt.show()	
