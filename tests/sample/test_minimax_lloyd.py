import numpy as np
import psdr
from scipy.linalg import orthogonal_procrustes 
from itertools import permutations

def test_minimax_lloyd_global_min(plot = False):
	# Here check if the algorithm converges close to the global minimizer
	# for a known case.
	# See JMY90, M = 7 case

	np.random.seed(2)

	m = 2
	M = 7
	domain = psdr.BoxDomain(0*np.ones(m), np.ones(m))
	Xhat_true = np.array([
			[0.5, 0.5], 
			[0.5, 0], [0.5,1],
			[1/3 - np.sqrt(7)/12, 3/4],
			[1/3 - np.sqrt(7)/12, 1/4],
			[2/3 + np.sqrt(7)/12, 1/4],
			[2/3 + np.sqrt(7)/12, 3/4],
			])
	Xhat = psdr.minimax_lloyd(domain, M, maxiter =  100, Xhat = None, xtol = 1e-9)

	Xhat_best = None
	best_score = np.inf
	
	for perm in permutations(range(len(Xhat))):
		perm = np.array(perm, dtype = np.int)
		R, scale = orthogonal_procrustes(Xhat, Xhat_true[perm])
		err = np.linalg.norm(Xhat @ R - Xhat_true[perm], 'fro')
		if err < best_score:
			best_score = err
			Xhat_best = Xhat @ R
		
	# TODO: need to align points out of Xhat

	print("Error in fit", best_score)

	if plot:
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		ax.plot(Xhat_best[:,0], Xhat_best[:,1],'rx')
		#XhatR = Xhat @ R
		#ax.plot(XhatR[:,0], XhatR[:,1],'ro')
		ax.plot(Xhat_true[:,0], Xhat_true[:,1],'k.')
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.axis('equal')
	
		plt.show()	

def test_minimax_lloyd(m = 2, M = 7, plot = False):
	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
	Xhat = psdr.minimax_lloyd(domain, M, maxiter =  100)

	if plot:
		import matplotlib.pyplot as plt
		fig, ax = plt.subplots()
		ax.plot(Xhat[:,0], Xhat[:,1],'k.')
		ax.set_xlim(-1,1)
		ax.set_ylim(-1,1)
		ax.axis('equal')
	
		plt.show()	

if __name__ == '__main__':
	test_minimax_lloyd_global_min(plot = True)
