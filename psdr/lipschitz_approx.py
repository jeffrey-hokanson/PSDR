import numpy as np
import cvxpy as cp
from .lipschitz import LipschitzMatrix
from .ridge import RidgeFunction
from .misc import merge


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
	def __init__(self, L, norm = 1):
		self._L = np.copy(L)
		self.norm = norm

		# Compute the active subspace
		U, s, VT = np.linalg.svd(L, full_matrices = False)
		self._U = VT.T

	def fit(self, X, fX):
		r""" 

		""" 
		self._X = np.copy(X)
		self._fX = lipschitz_approximation_compatible_data(self.L, X, fX, norm = self.norm) 
		self.epsilon = np.max(np.abs(self.fX - fX))	

	def eval(self, X):
		lb, ub = self.uncertainty(self.X, self.fX, X)
		return 0.5*(lb + ub)

