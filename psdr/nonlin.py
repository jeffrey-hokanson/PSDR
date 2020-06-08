import numpy as np
import cvxpy as cp
from .lipschitz_utils import _vec_norm

def measure_nonlinearity(X, fX, **kwargs):
	r""" Measure the nonlinearity of a function

	Parameters
	----------
	X: array-like (M,m)
		locations where samples are taken
	fX: array-like (M,)
		function values corresponding to X
	**kwargs:
		Arguments to be passed to a CVXPY problem.solve method

	Returns
	-------
	N: float
		Nonlinearity measure
	a: np.array (m,)
		Coefficients corresponding best linear fit

	Notes
	-----
	* Of the open source solvers distributed with CVXPY, OSQP will sometimes fail and SCS is slow;
	  ECOS seems to yield the best performance and so is chosen as the default solver unless overridden.
	  Because we only need to solve an LP, GUROBI is recommended if avalible.
	"""
	# Normalize input data formats
	X = np.array(X)
	fX = np.array(fX)

	if 'solver' not in kwargs:
		kwargs['solver'] = 'ECOS'	

	M, m = X.shape
	i, j = np.triu_indices(M, k = 1)
	P = X[i] - X[j]
	Pnorm = np.sqrt(_vec_norm(P))
	fdiff = (fX[i] - fX[j]).flatten()

	# Compute standard Lipschitz constant
	L0 = np.max(np.abs(fdiff)/Pnorm)

	# Compute Lipschitz constant when subtracting 	
	a = cp.Variable(m)
	L = cp.Variable(1)
	constraints = [ 
		fdiff + P @ a <= L * Pnorm,
		fdiff + P @ a >= -L * Pnorm 
	]
	prob = cp.Problem(cp.Minimize(L), constraints)
	prob.solve(**kwargs)

	return float(L.value/L0), a.value

