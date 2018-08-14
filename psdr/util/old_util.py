from __future__ import division
import numpy as np
from scipy.optimize import linprog as sp_linprog

# checking to see if system has gurobi installed
try:
	HAS_GUROBI = True
	import gurobipy as gpy
except ImportError, e:
	HAS_GUROBI = False

try:
	HAS_CVXOPT = True
	import cvxopt
except ImportError, e:
	HAS_CVXOPT = False


class LinProgException(Exception):
	def __init__(self, x = None, A_ub = None, b_ub = None, lb = None, ub = None, A_eq = None, b_eq = None):
		pass

class InfeasibleConstraints(LinProgException):
	pass

def clencurt(n, a = -1, b = 1):
	""" Return the Clenshaw-Curtis quadtrature nodes and weights on [a,b]

	Modified from Trefethen's clencurt.m code and Embree's clencurtab.m code
	"""

	theta = np.pi*np.arange(0,n+1)/n
	x = np.cos(theta)
	w = np.zeros(n+1)
	v = np.ones(n-1)
	if n % 2 == 0:
		w[0] = 1/(n**2 - 1)
		w[n] = w[0]
		for k in range(1, n//2):
			v -= 2 * np.cos( 2 * k * theta[1:n] ) / (4 * k**2 - 1)
		v -= np.cos(n * theta[1:n])/(n**2 - 1)
	else:
		w[0] = 1/n**2
		w[n] = w[0]
		for k in range(1, (n-1)//2 + 1):
			v -= 2 * np.cos( 2 * k * theta[1:n] )/ (4 * k**2 - 1)
	
	w[1:n] = 2*v/n
	
	x = a + (b - a)/2 * (1 + x)
	w = (b - a)/2 * w
	return x, w

def gauss_hermite(n):
	return np.polynomial.hermite.hermgauss(n)


def check_linprog_solution(x, A_ub = None, b_ub = None, lb = None, ub = None, A_eq = None, b_eq = None, tol = 1e-7):
	if A_ub is not None and b_ub is not None:
		err = np.dot(A_ub, x) - b_ub
		if not np.all(err < tol):
			print "Inequalitites failed", err
			return False

	if A_eq is not None and b_eq is not None:
		err = np.abs(np.dot(A_eq, x) - b_eq)
		if not np.all(err < tol):
			print "Equalities failed", err
			return False

	if lb is not None:
		if not np.all( x >= lb - tol):
			print "Lower bound failed", x - lb + tol
			return False
	 
	if ub is not None:
		if not np.all( x <= ub + tol):
			print "Upper bound failed", ub + tol - x
			return False

	return True





def projected_closest_point(A, b, A_ub = None, b_ub = None, A_eq = None, b_eq = None, lb = None, ub = None):
	""" Compute the closest point in the projected space

		min_x \| A x - b \|_2^2
		s.t. lb <= x <= ub
			 A_ub x <=  b_ub
			 A_eq x = b_eq
	"""
	
	return projected_closest_point_guroibi(A, b, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, lb = lb, ub = ub)

def projected_closest_point_guroibi(A, b, A_ub = None, b_ub = None, A_eq = None, b_eq = None, lb = None, ub = None):
	m = A.shape[1]
	model = gpy.Model()
	model.setParam('OutputFlag', 0)
	model.setParam('NumericFocus', 3)	# improve handeling of numerical instabilities
	
	# Determine the bounds
	if lb is None:
		lb = -np.inf * np.ones(m)
	if ub is None:
		ub = np.inf * np.ones(m)
	
	# Setup the variables
	x = []
	for i in range(m):
		if np.isfinite(lb[i]):
			lb_ = lb[i]
		else:
			lb_ = -gpy.GRB.INFINITY
		if np.isfinite(ub[i]):
			ub_ = ub[i]
		else:
			ub_ = gpy.GRB.INFINITY
		x.append(model.addVar(lb = lb_, ub = ub_, vtype = gpy.GRB.CONTINUOUS))
	model.update()	

	# Add equality constraints:
	if A_eq is not None and A_eq.shape[0] > 0:
		for A_, b_ in zip(A_eq, b_eq):
			expr = gpy.LinExpr()
			for i in range(m):
				 expr += A_[i]*x[i]
			model.addConstr(expr, gpy.GRB.EQUAL, b_)
	model.update()	

	# Add inequality constraints
	if A_ub is not None and A_ub.shape[0] > 0:
		for A_, b_ in zip(A_ub, b_ub):
			expr = gpy.LinExpr()
			for i in range(m):
				 expr += A_[i]*x[i]
			model.addConstr(expr, gpy.GRB.LESS_EQUAL, b_)
	model.update()	
		
	# Setup the objective
	obj = gpy.QuadExpr()
	for j in range(A.shape[0]):
		expr = gpy.LinExpr()
		for k in range(A.shape[1]):
			expr += A[j,k]*x[k] 
		expr -= b[j]
		obj += expr*expr
	
	model.setObjective(obj)
	model.update()

	model.optimize()
	
	if model.status == gpy.GRB.OPTIMAL:
		return np.array(model.getAttr('x', x)).reshape((m,))
	else:
		raise Exception('Gurobi did not solve the LP. Blame Gurobi.')
	



def closest_point(x0, L = None, A_ub = None, b_ub = None, A_eq = None, b_eq = None, lb = None, ub = None):
	if L is None:
		L = np.eye(x0.shape[0])
	# TODO: Implement other drivers
	U, s, VT = np.linalg.svd(L)
	if np.min(s) < 1e-7:
		L = L + 1e-7*np.eye(L.shape[0])
	return closest_point_guroibi(x0, L = L, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, lb = lb, ub = ub)

def closest_point_guroibi(x0, L = None, A_ub = None, b_ub = None, A_eq = None, b_eq = None, lb = None, ub = None):
	"""
		Solve 

			min_x \| L(x - x0)\|
			s.t.  A_ub x <= b_ub
				  A_eq x = b_eq
                  lb <= x <= ub
	"""

	m = x0.shape[0]

	if L is None:
		L = np.eye(m)

	model = gpy.Model()
	model.setParam('OutputFlag', 0)
	model.setParam('NumericFocus', 3)	# improve handeling of numerical instabilities
	
	if lb is None:
		lb = -np.inf * np.ones(m)
	if ub is None:
		ub = np.inf * np.ones(m)
	
	# Setup the variables
	x = []
	for i in range(m):
		if np.isfinite(lb[i]):
			lb_ = lb[i]
		else:
			lb_ = -gpy.GRB.INFINITY
		if np.isfinite(ub[i]):
			ub_ = ub[i]
		else:
			ub_ = gpy.GRB.INFINITY
		x.append(model.addVar(lb = lb_, ub = ub_, vtype = gpy.GRB.CONTINUOUS))
	model.update()	

	# Add equality constraints:
	if A_eq is not None and A_eq.shape[0] > 0:
		for A_, b_ in zip(A_eq, b_eq):
			expr = gpy.LinExpr()
			for i in range(m):
				 expr += A_[i]*x[i]
			model.addConstr(expr, gpy.GRB.EQUAL, b_)
	model.update()	

	# Add inequality constraints
	if A_ub is not None and A_ub.shape[0] > 0:
		for A_, b_ in zip(A_ub, b_ub):
			expr = gpy.LinExpr()
			for i in range(m):
				 expr += A_[i]*x[i]
			model.addConstr(expr, gpy.GRB.LESS_EQUAL, b_)
	model.update()	
		
	# Setup the objective
	obj = gpy.QuadExpr()
	for j in range(m):
		expr = gpy.LinExpr()
		for k in range(m):
			expr += L[j,k]*(x[k] - x0[k])	
		obj += expr*expr
	
	model.setObjective(obj)
	model.update()

	model.optimize()
	
	if model.status == gpy.GRB.OPTIMAL:
		return np.array(model.getAttr('x', x)).reshape((m,))
	else:
		raise Exception('Gurobi did not solve the LP. Blame Gurobi.')




if __name__ == '__main__':
	c = np.array([-4,-5])
	A = np.array([[2, 1, -1, 0],[1, 2, 0, -1]]).T
	b = np.array([3,3,0,0])
	sol = linprog_cvxopt(c, A_ub = A, b_ub = b)
	print sol
