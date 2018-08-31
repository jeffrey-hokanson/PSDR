import numpy as np
import gurobipy as gpy
from shared import *

def linprog_gurobi(c, A_ub = None, b_ub = None, lb = None, ub = None, A_eq = None, b_eq = None,
	feastol = 1e-8, opttol = 1e-8):

	# Clean up posing
	if b_eq is not None:
		b_eq = np.atleast_1d(np.array(b_eq))

	model = gpy.Model()
	model.setParam('OutputFlag', 0)		# Disable logging
	model.setParam('NumericFocus', 3)	# improve handeling of numerical instabilities
	
	# Increase convergence tolerances
	model.setParam('FeasibilityTol', feastol)
	model.setParam('OptimalityTol', opttol)
	
	n = c.shape[0]	

	# Add variables to model
	vars_ = []
	if lb is None:
		lb = -np.inf * np.ones(n)
	if ub is None:
		ub = np.inf * np.ones(n)

	for j in range(n):
		if np.isfinite(lb[j]):
			lb_ = lb[j]
		else:
			lb_ = -gpy.GRB.INFINITY

		if np.isfinite(ub[j]):
			ub_ = ub[j]
		else:
			ub_ = gpy.GRB.INFINITY
		vars_.append(model.addVar(lb=lb_, ub=ub_, vtype=gpy.GRB.CONTINUOUS))

	model.update()

	# Populate linear constraints
	if A_ub is not None and A_ub.shape[0] > 0:
		for i in range(A_ub.shape[0]):
			expr = gpy.LinExpr()
			for j in range(n):
				expr += A_ub[i,j]*vars_[j]
			model.addConstr(expr, gpy.GRB.LESS_EQUAL, b_ub[i])
	
	# Add equality constraints
	if A_eq is not None and A_eq.shape[0] > 0:
		m_eq, n_eq = A_eq.shape
		for i in range(m_eq):
			expr = gpy.LinExpr()
			for j in range(n_eq):
				expr += A_eq[i,j]*vars_[j]
			model.addConstr(expr, gpy.GRB.EQUAL, b_eq[i])

	# Populate objective
	obj = gpy.LinExpr()
	for j in range(n):
		obj += c[j]*vars_[j]
	model.setObjective(obj)
	model.update()

	# Solve
	model.optimize()

	if model.status == gpy.GRB.OPTIMAL:
		x_opt = np.array(model.getAttr('x', vars_)).reshape((n,))
		return x_opt
	elif model.status == gpy.GRB.INFEASIBLE:
		raise InfeasibleConstraints 
	else:
		raise LinProgException
