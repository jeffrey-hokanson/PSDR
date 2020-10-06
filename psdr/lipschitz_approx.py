import numpy as np
import cvxpy as cp
from .lipschitz import LipschitzMatrix
from .ridge import RidgeFunction
from .misc import merge
from .sample import maximin_design_1d
from .pgf import PGF


DEFAULT_CVXPY_KWARGS = {
	'solver': 'ECOS',
}


def lipschitz_approximation_compatible_data(L, X, fX, norm = 1, **kwargs):
	r""" Compute data that best approximates fX compatible with the Lipschitz matrix

	"""
	X = np.array(X)
	fX = np.array(fX)
	L = np.array(L)

	# Check data consistency
	assert len(fX.shape) == 1, "Input data must be one-dimensional"
	assert len(X) == len(fX), "Number of input coordinates do not match"
	assert X.shape[1] == L.shape[1], "Dimension of the Lipschitz matrix does not match the dimension of the space"

	# Set default arguments
	kwargs = merge(DEFAULT_CVXPY_KWARGS, kwargs)

	y = cp.Variable(len(fX))
	y.value = fX
	obj = cp.norm(y - fX, norm)

	I, J = np.tril_indices(len(X), k=1)

	# np.linalg.norm(L @ (X[i] - X[j]), 2)
	rhs = np.sqrt(np.sum( (L @ (X[I] - X[J]).T)**2, axis = 0))
	constraints = [cp.abs(y[I] - y[J]) <= rhs]

	prob = cp.Problem(cp.Minimize(obj), constraints)
	prob.solve(**kwargs)

	return y.value	


class LipschitzApproximation(LipschitzMatrix, RidgeFunction):
	def __init__(self, L, domain, norm = 1):
		self._L = np.copy(L)
		self.norm = norm
		self.domain = domain

		# Compute the active subspace
		U, s, VT = np.linalg.svd(L, full_matrices = False)
		self._U = VT.T
		self._dimension = self._U.shape[0]

	def fit(self, X, fX):
		r""" 

		""" 
		self._X = np.copy(X)
		self._fX = np.copy(fX)
		assert np.all(self.domain.isinside(X)), "Samples outside of the domain"
		self._y = lipschitz_approximation_compatible_data(self.L, X, fX, norm = self.norm) 
		self.epsilon = np.max(np.abs(self._y - self._fX))	

	def eval(self, X):
		lb, ub = self.uncertainty(self._X, self._y, X)
		return 0.5*(lb + ub)


	def shadow_plot(self, dim = 1, ax = 'auto', pgfname = None, nsamp = 200, U = None):
		
		if ax == 'auto':
			import matplotlib.pyplot as plt
			if dim == 1:
				fig, ax = plt.subplots(figsize = (6,6))
			else:
				# Hack so that plot is approximately square after adding colorbar 
				fig, ax = plt.subplots(figsize = (7.5,6))
		
		if U is None:
			U = self.U[:,:dim]

		if dim == 1:
			assert U.shape[1] == 1, "Must be a one-dimensional ridge function"

			# Plot the central estimate and uncertainty
			X = maximin_design_1d(self.domain, nsamp, L = self.L) 
			
			lb, ub = self.uncertainty(self._X, self._y, X)
			c = 0.5*(lb+ub)
			xx = (U.T @ X.T).flatten()
		
			if ax is not None:
				
				# points
				ax.plot( (U.T @ self.X.T).flatten(), self._fX,'k.')
				# central approximation
				ax.plot(xx, c, 'r')
				# uncertainty
				ax.fill_between(xx, lb, ub, alpha = 0.3, color = 'blue')
	
			if pgfname is not None:
				pgf = PGF()
				pgf.add('x', xx)
				pgf.add('lb', lb)
				pgf.add('c', c)
				pgf.add('ub', ub)
				pgf.write(pgfname + '_curves.dat')

				pgf = PGF()
				pgf.add('x', (U.T @ self.X.T).flatten())
				pgf.add('y', self._fX)
				pgf.write(pgfname + '_points.dat')	 			
			
			return ax		
		

		 
