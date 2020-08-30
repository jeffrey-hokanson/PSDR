import numpy as np
import psdr
import seaborn as sns
from psdr.pgf import PGF

def sine_func(x):
	return np.sin(np.pi/5*np.sum(x))

m = 10
domain = psdr.BoxDomain(-np.ones(m), np.ones(m))

fun = psdr.Function(sine_func, domain)

L = np.pi/1000* np.outer(np.ones(m), np.ones(m))

M = 20

# Ridge subspace
u = np.ones((m,1))/np.sqrt(m)

proj_domain = psdr.BoxDomain([-np.sqrt(m)], [np.sqrt(m)])


import matplotlib.pyplot as plt

# Get data for drawing plot
XX = np.linspace(-np.sqrt(m),np.sqrt(m),1000).reshape(-1,1) @ u.T
fXX = fun(XX)
fig, ax = plt.subplots(1,2)
ax[0].plot( (u.T @ XX.T).flatten(), fXX)

pgf = PGF()
pgf.add('x', (u.T @ XX.T).flatten())
pgf.add('y', fXX)
pgf.write('data/fig_sine_fun.dat')

# plot minimax design
X = psdr.minimax_design_1d(fun.domain, M, L = L[0,:].reshape(1,-1))
minimax_dist = psdr.fill_distance(proj_domain, (u.T @ X.T).T)
ax[0].plot( (u.T @ X.T).flatten(), fun(X), 'r.')

pgf = PGF()
pgf.add('x', (u.T @ X.T).flatten())
pgf.add('y', fun(X))
pgf.write('data/fig_sine_minimax.dat')



# Plot a random design
np.random.seed(0)

X = psdr.random_sample(fun.domain, M)
ax[0].plot( (u.T @ X.T).flatten(), fun(X), 'g.')

pgf = PGF()
pgf.add('x', (u.T @ X.T).flatten())
pgf.add('y', fun(X))
pgf.write('data/fig_sine_rand.dat')

# Plot an LHS
np.random.seed(0)
X = psdr.latin_hypercube_maximin(fun.domain, M, maxiter = 1)
ax[0].plot( (u.T @ X.T).flatten(), fun(X), 'k.')

pgf = PGF()
pgf.add('x', (u.T @ X.T).flatten())
pgf.add('y', fun(X))
pgf.write('data/fig_sine_lhs.dat')


# Generate a bunch of LHS designs
lhs_dist = []
for i in range(100):
	np.random.seed(i)
	X = psdr.latin_hypercube_maximin(fun.domain, M, maxiter = 1)
	lhs_dist.append(psdr.fill_distance(proj_domain, (u.T @ X.T).T))

rand_dist = []
for i in range(100):
	np.random.seed(i)
	X = psdr.random_sample(fun.domain, M)
	rand_dist.append(psdr.fill_distance(proj_domain, (u.T @ X.T).T))

sns.swarmplot(data = [rand_dist, lhs_dist,], orient = 'v', ax = ax[1])

for coll, name in zip(ax[1].collections, ['rand', 'lhs']):
	x, y = np.array(coll.get_offsets()).T
	pgf = PGF()
	pgf.add('x', x)
	pgf.add('y', y)
	pgf.write('data/fig_sine_swarm_%s.dat' % name)

# 
pgf = PGF()
pgf.add('x',[0])
pgf.add('y', [minimax_dist])
pgf.write('data/fig_sine_swarm_minimax.dat')

plt.show()

