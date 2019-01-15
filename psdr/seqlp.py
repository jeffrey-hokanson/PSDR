""" Sequential Linear Program
"""

import numpy as np
import cvxpy as cp
import warnings
from gn import trajectory_linear

def sequential_lp(f, x0, jac, search_constraints = None,
	norm = 2, trajectory = trajectory_linear, obj_lb = None, obj_ub = None, 
	maxiter = 100, bt_maxiter = 20, domain = None,
	tol_dx = 1e-10, **kwargs):
	r""" Solves a nonlinear optimization by sequential least squares


	Given the optimization problem

	.. math::

		\min{\mathbf{x} \in \mathbb{R}^m} &\ \| \mathbf{f}(\mathbf{x}) \|_p \\
		\text{such that} & \  \text{lb} \le \mathbf{f} \le \text{ub}

	this function solves this problem by linearizing both the objective and constraints
	and solving a sequence of linear programs.

	"""

	assert norm in [1,2,np.inf, None], "Invalid norm specified."

	if search_constraints is None:
		search_constraints = lambda x, p: []
	
	if domain is None:
		domain = UnboundedDomain(len(x0))


	# Start optimizaiton loop
	x = np.copy(x0)
	fx = np.array(f(x))

	if norm in [1,2, np.inf]
		objfun = lambda fx: np.linalg.norm(fx, ord = norm)
	else:
		objfun = lambda fx: float(fx)

	objval = objfun(fx)
	
	if verbose:
		print 'iter |     objective     |  norm px |  bt step | TR radius |'
		print '-----|-------------------|----------|----------|-----------|'
		print '%4d | %+14.10e |          |          |           |' % (0, objval) 

	for it in range(maxiter):
		jacx = jac(x)
	
		# Search direction
		p = cp.Variable(len(x))

		# Linearization of the objective function
		f_lin = fx + p.__rmatmul__(jacx)

		if norm == 1: obj = cp.norm1(f_lin)
		elif norm == 2: obj = cp.norm2(f_lin)
		elif norm == np.inf: obj = cp.norm_inf(f_lin)
		elif norm == None: obj = f_lin

		# Now setup constraints
		constraints = []

		# First, constraints on "f"
		if obj_lb is not None:
			constraints.append(obj_lb <= f_lin)
		if obj_ub is not None:
			constraints.append(f_lin <= obj_ub)  
		

		# Constraints on the search direction specified by user
		constraints += search_constraints(x, p)

		# Append constraints from the domain of x
		constraints += domain._build_constraints(px - x)

		# TODO: Add additional user specified constraints following scipy.optimize.NonlinearConstraint (?)


		# Solve for the search direction
		with warnings.catch_warnings():
			warnings.simplefilter('ignore', PendingDeprecationWarning)
			problem = cp.Problem(cp.Minimize(obj), constraints)
			problem.solve(**kwargs)
		
		if problem.status in ['infeasible', 'unbounded']:
			raise Exception(problem.status)

		px = p.value		

		alpha = 1.
		stop = False
		for it2 in range(bt_maxiter):
			x_new = trajectory(x, px, alpha)

			if np.all(np.isclose(x, x_new, rtol = tol_dx, atol = 0)):
				stop = True
				break

			fx_new = np.array(f(x_new))
			objval_new = objfun(fx_new)

			#if objval_new <= objval + c_armijo*alpha
			

