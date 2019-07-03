from __future__ import print_function

import numpy as np
from scipy.linalg import orth
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import psdr 
from psdr.pgf import PGF


def plot_projection(X, L, center, ax, pgfname, stretch, **kwargs):
	U = orth(np.atleast_2d(L).T)
	P = np.outer(U,U.T)

	center = center - P.dot(center)

	# Line we project onto	
	line = np.vstack([
		center - stretch*L,
		center + stretch*L,
		])
	ax.plot(line[:,0], line[:,1], **kwargs)
	pgf = PGF()
	pgf.add('x', line[:,0])
	pgf.add('y', line[:,1])
	pgf.write(pgfname + '_line.dat')

	# projection lines
	lines = []		# lines from points to the projection axis
	dots = []		# dots on the projection axis
	for x in X:
		lines.append(x)
		lines.append(center + P.dot(x))
		dots.append(lines[-1])
		lines.append(np.nan*np.ones(2))
	lines = np.vstack(lines)
	ax.plot(lines[:,0], lines[:,1], ':', **kwargs)
	dots = np.vstack(dots)
	ax.plot(dots[:,0], dots[:,1], '.', **kwargs)

	pgf = PGF()
	pgf.add('x', lines[:,0])
	pgf.add('y', lines[:,1])
	pgf.write(pgfname + '_lines.dat')
	
	pgf = PGF()
	pgf.add('x', dots[:,0])
	pgf.add('y', dots[:,1])
	pgf.write(pgfname + '_dots.dat')

if __name__ == '__main__':
	np.random.seed(0)

	dom = psdr.BoxDomain(-np.ones(2), np.ones(2))

	fig, axes = plt.subplots(1, 2, figsize = (10, 5))

	# Number of samples
	M = 20

	# Latin Hypercube sampling
	X = dom.latin_hypercube(M)

	pgf = PGF()
	pgf.add('x', X[:,0])
	pgf.add('y', X[:,1])
	pgf.write('data/fig_latin_lhs_sample.dat')


	ax = axes[0]
	ax.plot(X[:,0], X[:,1], 'k.')
	ax.set_title('Latin Hypercube')

	L = np.array([[1,0]])
	center = np.array([0,-1.3])
	plot_projection(X, L, center, ax, 'data/fig_latin_lhs_hor', stretch = 1.2, color = 'b')
	
	L = np.array([[0,1]])
	center = np.array([1.3,0])
	plot_projection(X, L, center, ax, 'data/fig_latin_lhs_vert', stretch = 1.2, color = 'r')


	# Lipschitz based sampling
	ax = axes[1]
	#Ls = [orth(np.ones((2,1))).T, np.array([[1],[0]]).T]
	#Ls = [np.array([[0],[1]]).T, np.array([[1],[0]]).T]
	Ls = [np.array([[2,1]]), np.array([[1, 2]])]
	X = psdr.lipschitz_sample(dom, M, Ls = Ls, verbose = True, maxiter = 1000, jiggle = False, maxiter_maximin = 0)
	pgf = PGF()
	pgf.add('x', X[:,0])
	pgf.add('y', X[:,1])
	pgf.write('data/fig_latin_lip_sample.dat')
	
	ax.plot(X[:,0], X[:,1], 'k.')
	ax.set_title('Lipschitz')

	L = Ls[0]
	center = np.array([1.1,-1.1])
	plot_projection(X, L, center, ax, 'data/fig_latin_lip0', stretch = 1.2, color = 'r')
	
	L = Ls[1]
	center = np.array([1.2,-1])
	plot_projection(X, L, center, ax, 'data/fig_latin_lip0', stretch = 1.2, color = 'b')

	
	for ax in axes:
		ax.set_xlim(-1.5,2.5)
		ax.set_ylim(-2.5,1.5)
		ax.add_patch(Rectangle((-1,-1), 2, 2, alpha = 1, fill = None, edgecolor = 'black'))
		ax.axis('off')
	plt.show()

