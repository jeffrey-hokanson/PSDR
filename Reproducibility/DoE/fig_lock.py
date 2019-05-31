from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import psdr 
from psdr.pgf import PGF
from fig_latin import plot_projection

if __name__ == '__main__':
	# The locking phenomina, where given enough Lipschitz matrices of sufficient rank to 
	# uniquely specify a point if all samples where 
	np.random.seed(0)	
	dom = psdr.BoxDomain(-np.ones(2), np.ones(2))

	fig, axes = plt.subplots(1, 2, figsize = (10, 5))
	M = 20
	# First 9 lock	
	Ls = [np.array([[2,1]]), np.array([[1, 2]])]
	
	for ax, depth in zip(axes, [2, 1]):
		X = []
		for i in range(M):
			x = psdr.seq_maximin_sample(dom, X, Ls = Ls, depth = depth, Nsamp = int(1e3))
			X.append(x)
		X = np.vstack(X)
		pgf = PGF()
		pgf.add('x', X[:,0])
		pgf.add('y', X[:,1])
		pgf.write('data/fig_lock_d%d_sample.dat' % depth)
		
		ax.plot(X[:,0], X[:,1], 'k.')
		ax.set_title('depth=%d' % depth)
		centers = [np.array([1.1,-1.1]), np.array([2, 0])]
		colors = ['b', 'r']
		for i, (L, center, color) in enumerate(zip(Ls, centers, colors)):
			plot_projection(X, L, center, ax, 'data/fig_lock_d%d_L%d' % (depth, i) , stretch = 1.2, color = color)

	for ax in axes:
		ax.set_xlim(-1.5,2.5)
		ax.set_ylim(-2.5,1.5)
		ax.add_patch(Rectangle((-1,-1), 2, 2, alpha = 1, fill = None, edgecolor = 'black'))
		ax.axis('off')
	plt.show()


