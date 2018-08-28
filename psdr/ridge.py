"""Ridge function approximation from function values"""
# (c) 2017 Jeffrey M. Hokanson (jeffrey@hokanson.us)

import numpy as np


class RidgeApproximation(object):
	
	def predict(self, Xnew):
		raise NotImplementedError

	def predict_ridge(self, Ynew):
		raise NotImplementedError
	
	def plot(self, axes = None, X = None, y = None, domain = None):
		from matplotlib import pyplot as plt
		if X is None or y is None:
			X = self.X
			y = self.y
		
		if axes is None:
			fig, axes = plt.subplots(figsize = (6,6))

		if self.subspace_dimension == 1:
			Y = np.dot(self.U.T, X.T).flatten()
			lb = np.min(Y)
			ub = np.max(Y)
			
			axes.plot(Y, y, 'k.', markersize = 6)
			xx = np.linspace(lb, ub, 100)
			XX = np.array([self.U.flatten()*x for x in xx])
			axes.plot(xx, self.predict(XX), 'r-', linewidth = 2)

			if domain is not None:
				ridge_domain = build_ridge_domain(domain, self.U)
				axes.axvspan(ridge_domain.lb[0], ridge_domain.ub[0], color = 'b', alpha = 0.15)

		elif self.subspace_dimension == 2:
			Y = np.dot(self.U.T, X.T).T
			# Construct grid
			x = np.linspace(np.min(Y[:,0]), np.max(Y[:,0]), 100)	
			y = np.linspace(np.min(Y[:,1]), np.max(Y[:,1]), 100)
			xx, yy = np.meshgrid(x, y)
			# Sample the ridge function
			UXX = np.vstack([xx.flatten(), yy.flatten()])
			XX = np.dot(self.U, UXX).T
			YY = self.predict(XX)
			YY = np.reshape(YY, xx.shape)
			
			axes.contour(xx, yy, YY, 
				levels = np.linspace(np.min(self.y), np.max(self.y), 20), 
				vmin = np.min(self.y), vmax = np.max(self.y),
				linewidths = 0.5)
			
			# Plot points
			axes.scatter(Y[:,0], Y[:,1], c = self.y, s = 3)

			# plot boundary
			if domain is not None:
				ridge_domain = build_ridge_domain(domain, self.U)	
				Y = ridge_domain.X
				for simplex in ridge_domain.hull.simplices:
					axes.plot(Y[simplex,0], Y[simplex,1], 'k-')
		else:
			raise NotImplementedError

		return axes

	def plot_pgf(self, base_name, X = None, y = None):
		if X is None or y is None:
			X = self.X
			y = self.y

		if self.subspace_dimension == 1:
			Y = np.dot(self.U.T, X.T).flatten()
			lb = np.min(Y)
			ub = np.max(Y)
		
			pgf = PGF()
			pgf.add('Ux', Y.flatten())
			pgf.add('fx', y) 
			pgf.write('%s_data.dat' % base_name)
			
			xx = np.linspace(lb, ub, 100)
			XX = np.array([self.U.flatten()*x for x in xx])
			pgf = PGF()
			pgf.add('Ux', xx)
			pgf.add('predict', self.predict(XX))
			pgf.write('%s_fit.dat' % base_name)
