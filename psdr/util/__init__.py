
# This should always load: Scipy is a manditory dependency
from wrapper_scipy import *
try:
	HAS_GUROBI = True
	from wrapper_gurobi import *
except ImportError:
	HAS_GUROBI = False

try:
	HAS_CVXOPT = True
	from wrapper_cvxopt import *
except ImportError, e:
	HAS_CVXOPT = False

from old_util import *

if HAS_GUROBI:
	DEFAULT_BACKEND = 'GUROBI'
elif HAS_CVXOPT:
	DEFAULT_BACKEND = 'CVXOPT'
else:
	DEFAULT_BACKEND = 'SCIPY'


def linprog(c, A_ub = None, b_ub = None, A_eq = None, b_eq = None,  lb = None, ub = None, 
	backend = DEFAULT_BACKEND, show_progress = False, **kwargs):
	""" Wrapper around several linear program solvers

	Solves
			min_x  c^T x
			where 
				A_ub x <= b_ub
				A_eq x = b_eq
				lb <= x <= ub

	"""

	assert backend in ['CVXOPT', 'GUROBI', 'SCIPY'], "Unknown backend requested"

	if backend == 'CVXOPT':
		assert HAS_CVXOPT, "CVXOPT not avalible on this system"
		x = linprog_cvxopt(c, A_ub = A_ub, b_ub = b_ub, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq,
			show_progress = show_progress, **kwargs)
	elif backend == 'GUROBI':
		assert HAS_GUROBI, "GUROBI not avalible on this system"
		x = linprog_gurobi(c, A_ub = A_ub, b_ub = b_ub, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq)
	else:
		x = linprog_scipy(c, A_ub = A_ub, b_ub = b_ub, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq)

	if check_linprog_solution(x, A_ub = A_ub, b_ub = b_ub, lb = lb, ub = ub, A_eq = A_eq, b_eq = b_eq):
		return x
	else:
		raise LinProgException

