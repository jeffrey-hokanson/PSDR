import numpy as np
import psdr


def test_minimax_voronoi(N = 7, m = 2):
	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))
	L = np.random.randn(m,m)
	L = np.eye(m)

	Xhat = psdr.minimax_voronoi(domain, N, L = L, maxiter = 500)
	radius = 0

	if True:
		import matplotlib.pyplot as plt
		from matplotlib.patches import Polygon, Rectangle
		from matplotlib.collections import PatchCollection
		fig, ax = plt.subplots()

		# plot of the domain
		ax.add_patch(Rectangle((domain.lb[0], domain.lb[1]), domain.ub[0] - domain.lb[0], domain.ub[1] - domain.lb[1], fc = 'r', ec = 'r' ))
		ax.set_xlim(domain.lb[0]-0.2, domain.ub[0]+0.2)
		ax.set_ylim(domain.lb[1]-0.2, domain.ub[1]+0.2)

		# Epsilon ball in L metric
#		th = 0j*np.linspace(0,2*np.pi)[0:-1]
#		Z = radius*np.vstack([np.exp(th).real, np.exp(th).imag]).T
#		LZ = (np.linalg.inv(L).T @ Z.T).T

#		for x in Xhat:
#			ax.add_patch(Polygon(LZ + np.outer(np.ones(LZ.shape[-1]), x), fc = 'b', alpha = 0.3))		

		ax.plot(Xhat[:,0], Xhat[:,1], 'k.')
		ax.axis('equal')
		plt.show()

if __name__ == '__main__':
	test_minimax_voronoi()
