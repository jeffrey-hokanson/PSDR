""" Sequential Linear Program
"""

import numpy as np
import cvxpy as cp
from domains import UnboundedDomain
import warnings
from gn import trajectory_linear

def sequential_lp(f, x0, jac, search_constraints = None,
	norm = 2, trajectory = trajectory_linear, obj_lb = None, obj_ub = None, 
	maxiter = 100, bt_maxiter = 50, domain = None,
	tol_dx = 1e-10, c_armijo = 0.01,  verbose = False, **kwargs):
	r""" Solves a nonlinear optimization by sequential least squares


	Given the optimization problem

	.. math::

		\min{\mathbf{x} \in \mathbb{R}^m} &\ \| \mathbf{f}(\mathbf{x}) \|_p \\
		\text{such that} & \  \text{lb} \le \mathbf{f} \le \text{ub}

	this function solves this problem by linearizing both the objective and constraints
	and solving a sequence of linear programs.

	
	References
	----------
	.. FS89


	"""
	
	if 'solver' not in kwargs:
		kwargs['solver'] = 'ECOS'

	assert norm in [1,2,np.inf, None], "Invalid norm specified."

	if search_constraints is None:
		search_constraints = lambda x, p: []
	
	if domain is None:
		domain = UnboundedDomain(len(x0))



	if norm in [1,2, np.inf]:
		objfun = lambda fx: np.linalg.norm(fx, ord = norm)
	else:
		objfun = lambda fx: float(fx)

	# evalutate KKT norm
	def kkt_norm(fx, jacx):
		kkt_norm = np.nan
		if norm == np.inf:
			# TODO: allow other constraints into the solution
			t = objfun(fx)
			obj_grad = np.zeros(len(x)+1)
			obj_grad[-1] = 1.
			con = np.hstack([fx - t, -fx -t])
			con_grad = np.zeros((2*len(fx),len(x)+1))
			con_grad[:len(fx),:-1] = jacx
			con_grad[:len(fx),-1] = -1.
			con_grad[len(fx):,:-1] = -jacx
			con_grad[len(fx):,-1] = -1.

			# Find the active constraints (which have non-zero Lagrange multipliers)
			I = np.abs(con) < 1e-10
			lam, kkt_norm = scipy.optimize.nnls(con_grad[I,:].T, -obj_grad)
		elif norm == 1.:
			t = np.abs(fx)
			obj_grad = np.zeros(len(x) + len(fx))
			obj_grad[len(x):] = 1.
			con = np.hstack([fx - t, -fx-t])
			con_grad = np.zeros((2*len(fx), len(x)+len(fx)))
			con_grad[:len(fx),:len(x)] = jacx
			con_grad[:len(fx),len(x):] = -1.
			con_grad[len(fx):,:len(x)] = -jacx
			con_grad[len(fx):,len(x):] = -1.
			I = np.abs(con) == 0.
			lam, kkt_norm = scipy.optimize.nnls(con_grad[I,:].T, -obj_grad)
		elif norm == 2.:
			kkt_norm = np.linalg.norm(jacx.dot(fx))
		# TODO: Should really orthogonalize against unallowed search directions
		#err = con_grad[I,:].T.dot(lam) + obj_grad
		#print err
	
		return kkt_norm

	# Start optimizaiton loop
	x = np.copy(x0)
	fx = np.array(f(x))
	objval = objfun(fx)
	jacx = jac(x)

	
	if verbose:
		print 'iter |     objective     |  norm px | TR radius | KKT norm |'
		print '-----|-------------------|----------|-----------|----------|'
		print '%4d | %+14.10e |          |           | %8.2e |' % (0, objval, kkt_norm(fx, jacx)) 

	Delta = 1.

	for it in range(maxiter):
	
		# Search direction
		p = cp.Variable(len(x))

		# Linearization of the objective function
		f_lin = fx + p.__rmatmul__(jacx)

		if norm == 1: obj = cp.norm1(f_lin)
		elif norm == 2: obj = cp.norm(f_lin)
		elif norm == np.inf: obj = cp.norm_inf(f_lin)
		elif norm == None: obj = f_lin

		# Now setup constraints
		constraints = []

		# First, constraints on "f"
		#if obj_lb is not None:
		#	constraints.append(obj_lb <= f_lin)
		#if obj_ub is not None:
		#	constraints.append(f_lin <= obj_ub)  
		

		# Constraints on the search direction specified by user
		constraints += search_constraints(x, p)

		# Append constraints from the domain of x
		constraints += domain._build_constraints(p - x)

		# TODO: Add additional user specified constraints following scipy.optimize.NonlinearConstraint (?)

		stop = False
		for it2 in range(bt_maxiter):
			trust_region_constraints = [cp.norm(p) <= Delta]
			# Solve for the search direction
			with warnings.catch_warnings():
				warnings.simplefilter('ignore', PendingDeprecationWarning)
				problem = cp.Problem(cp.Minimize(obj), constraints + trust_region_constraints)
				problem.solve(**kwargs)
			
			if problem.status in ['infeasible', 'unbounded']:
				raise Exception(problem.status)

			px = p.value	
	
			x_new = trajectory(x, px, 1.)

			# If we haven't moved that far, stop
			if np.all(np.isclose(x, x_new, rtol = tol_dx, atol = 0)):
				stop = True
				break

			# Evaluate value at new point
			fx_new = np.array(f(x_new))
			objval_new = objfun(fx_new)

			#if objval_new - objval <= c_armijo*(pred_objval_new - objval):
			if objval_new < objval:
				x = x_new
				fx = fx_new
				objval = objval_new
				Delta = max(1., Delta*2)
				break

			Delta *=0.5
		
		if it2 == bt_maxiter-1:
			stop = True

		# Update the jacobian information
		jacx = jac(x)

		if verbose:
			print '%4d | %+14.10e | %8.2e |  %8.2e | %8.2e |' % (it+1, objval, np.linalg.norm(px), Delta, kkt_norm(fx, jacx))
		if stop:
			break	

	return x
			
if __name__ == '__main__':
	from polyridge import *
	
	np.random.seed(3)
	p = 3
	m = 4
	n = 1
	M = 100

	norm = np.inf
	norm = 1
	U = orth(np.random.randn(m,n))
	coef = np.random.randn(len(LegendreTensorBasis(n,p)))
	prf = PolynomialRidgeFunction(LegendreTensorBasis(n,p), coef, U)

	X = np.random.randn(M,m)
	fX = prf.eval(X)

	pra = PolynomialRidgeApproximation(degree = p, subspace_dimension  = n, norm = norm, scale = True)

	def residual(U_c):
		r = pra._residual(X, fX, U_c)
		return r

	def jacobian(U_c):
		U = U_c[:m*n].reshape(m,n)
		pra.set_scale(X, U)
		J = pra._jacobian(X, fX, U_c)
		return J
	
	# Trajectory
	trajectory = lambda U_c, p, alpha: pra._trajectory(X, fX, U_c, p, alpha)

	def search_constraints(U_c, pU_pc):
	#	M, m = X.shape
	#	N = len(self.basis)
	#	n = self.subspace_dimension
		U = U_c[:m*n].reshape(m,n)
		constraints = [ pU_pc[k*m:(k+1)*m].__rmatmul__(U.T) == np.zeros(n) for k in range(n)]
		return constraints

		
	U0 = orth(np.random.randn(m,n))
	U0 = U
	c = np.random.randn(len(coef))
	pra.set_scale(X, U0)
	U_c0 = np.hstack([U0.flatten(), c])

	U_c = sequential_lp(residual, U_c0, jacobian, search_constraints, norm = norm, 
		trajectory = trajectory, verbose = True)
	print U_c
	print U	
