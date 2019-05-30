from __future__ import print_function, division
""" misc. utilities and defintions
"""

def merge(x, y):
	z = x.copy()
	z.update(y)
	return z

DEFAULT_CVXPY_KWARGS = {
	'solver': 'CVXOPT',
	'reltol': 1e-10,
	'abstol' : 1e-10,
	'verbose': False,
	'kktsolver': 'robust', 
	'warm_start': True,
}

class SolverError(ValueError):
	pass
