import numpy as np
import cvxopt
from shared import *

def linprog_cvxopt(c, A_ub = None, b_ub = None, lb = None, ub = None, A_eq = None, b_eq = None,
	 reltol = 1e-10, abstol = 1e-10, feastol = 1e-10, verbose = False):
	""" Solve a linear program using CVXOPT
	See linprog for documentation of arguments
	"""
	n = c.shape[0]
	c = cvxopt.matrix(c.reshape(n,1).astype(np.float64), (n,1))
	
	# Convert the inequality constraints
	if A_ub is not None and A_ub.shape[0] > 0:
		# Here we use the notation of CVXOPT documentation
		G = np.copy(A_ub)
		h = np.copy(b_ub)
	else:
		G = np.zeros((0,n))
		h = np.zeros((0,))
	
	# Since CVXOPT does not explicitly support bound constraints, add these here	
	if lb is not None:
		G = np.vstack([G, np.eye(n)])
		h = np.hstack([h, lb])
	if ub is not None:
		G = np.vstack([G, -np.eye(n)])
		h = np.hstack([h, -ub])
	
	# Convert these matrices to CVXOPT format
	G = cvxopt.matrix(G.astype(np.float64))
	h = cvxopt.matrix(h.astype(np.float64))

	# Convert the equality constraints
	if A_eq is not None and A_eq.shape[0] > 0:
		A = cvxopt.matrix(A_eq.astype(np.float64))
		b = cvxopt.matrix(b_eq.astype(np.float64))
	else:
		A = None
		b = None

	cvxopt.solvers.options['show_progress'] = verbose
	cvxopt.solvers.options['reltol'] = reltol
	cvxopt.solvers.options['abstol'] = abstol
	cvxopt.solvers.options['feastol'] = feastol
	sol = cvxopt.solvers.lp(c, G, h, A = A, b = b)
	if sol['status'] == 'optimal':
		return np.array(sol['x'])
	else:
		raise LinProgException
