
from __future__ import print_function
import numpy as np
import cvxpy as cp
import warnings

from .domains import UnboundedDomain
from .gn import trajectory_linear, linesearch_armijo


__all__ = ['minimax',
	]

def minimax(f, x0, domain = None, trajectory = trajectory_linear, 
	c_armijo = 0.01, maxiter = 100, trust_region = True, search_constraints = None,
	tol_dx = 1e-10, tol_df = 1e-10,
	verbose = False, bt_maxiter = 30, **kwargs):
	r""" Solves a minimax optimization problem via sequential linearization and line search

	Given a vector-valued function :math:`f: \mathcal{D}\subset\mathbb{R}^m\to \mathbb{R}^q`,
	solve the minimax optimization problem:

	.. math::
	
		\min_{\mathbf{x}\in \mathcal{D}}  \max_{i=1,\ldots, q} f_i(\mathbf{x}).

	This implements a minor modification of the algorithm described by Osborne and Watson [OW69]_.
	The basic premise is that we add in a slack variable :math:`t_k` representing the maximum value
	of :math:`f_i(\mathbf{x}_k)` at the current iterate :math:`\mathbf{x}_k`.
	Then we linearize the functions :math:`f_i` about this point, 
	yielding additional linear constraints: 
	

	.. math::
		:label: minimax1 

		\min_{\mathbf{p}_\mathbf{x} \in \mathbb{R}^m, p_t}  & \  p_t \\
		\text{such that} & \ t_k + p_t \ge f_i(\mathbf{x}_k) + 
			\mathbf{p}_\mathbf{x}^\top \nabla f_i(\mathbf{x}_k) \quad \forall i=1,\ldots,q

	This yields a search direction :math:`\mathbf{p}_\mathbf{x}`,
	along which we preform a line search to find a point satisfying the constraint:

	.. math::
		:label: minimax2

		\min_{\alpha \ge 0} \max_{i=1,\ldots,q} f_i(T(\mathbf{x}_k, \mathbf{p}_\mathbf{x}, \alpha))
			\le t_k + c\alpha p_t.

	Here :math:`T` represents the trajectory which defaults to a linear trajectory:

	.. math::
		
		T(\mathbf{x}, \mathbf{p}, \alpha) = \mathbf{x} + \alpha\mathbf{p}

	but, if provided can be more sophisticated. 
	The substantial difference from [OW69]_ is using an inexact backtracking linesearch
	is used to find the :math:`\alpha` satisfying the Armijo like condition :eq:`minimax2`;
	as originally proposed, Osborne and Watson use an exact line search. 

	Parameters
	----------
	f : BaseFuction-like
		Provides a callable interface f(x) that evalates the functions at x
		and access to the gradient by f.grad(x) 
	x0: array-like (m,)
		Starting point for optimization	
	domain: Domain, optional
		Add constraints to each step enforcing that steps remain in the domain
		(assumes a linear trajectory for steps)
	trajectory: callable, optional
		Function taking current location x, direction p, and step length alpha
		and returning a new point x.
	c_armijo: float, optional
		Coefficient used in the Armijo backtracking linesearch
	maxiter: int, optional
		Maximum number of iterations
	trust_region: bool, optional
		If true, enforce a spherical trust region on each step
	search_constraints: callable
		Function taking the current iterate x and cvxpy.Variable representing the search direction
		and returning a list of cvxpy constraints. 
	tol_dx: float
		convergence tolerance in terms of movement of x
	tol_df: float
		convergence tolerance in terms of the maximum
	verbose: bool
		If true, display convergence history 
	bt_maxiter: int
		Number of backtracking line search steps to take
	**kwargs: dict
		Additional parameters to pass to cvxpy.Problem.solve()

	Returns
	-------
	x: np.array (m,)
		Solution to minimax problem.

	
	References
	----------
	.. [OW69] Osborne and Watson, "An algorithm for minimax approximation in the nonlinear case",
		The Computer Journal, 12(1) 1969 pp. 63--68
		https://doi.org/10.1093/comjnl/12.1.63

	"""

	x0 = np.array(x0)

	if domain is None:
		domain = UnboundedDomain(len(x0))
	
	#assert isinstance(domain, Domain), "Must provide a domain for the space"
	assert domain.isinside(x0), "Starting point must be inside the domain"

	if search_constraints is None:
		search_constraints = lambda x, p: []


	if 'solver' not in kwargs:
		kwargs['solver'] = 'ECOS'


	# Start of optimization loop
	x = np.copy(x0)
	fx = np.array(f(x))
	t = np.max(fx)	
	if verbose:
		print('iter |     objective     |  norm px |  bt step | TR radius |')
		print('-----|-------------------|----------|----------|-----------|')
		print('%4d | %+14.10e |          |          |           |' % (0, t) )

	if trust_region:
		Delta = 1.
	else:
		Delta = np.nan

	for it in range(maxiter):
		gradx = np.array(f.grad(x))

		# Solve optimization problem for step
		pt = cp.Variable(1)
		px = cp.Variable(len(x))
		constraints = [ (t + pt)*np.ones(fx.shape[0]) >= fx + px.__rmatmul__(gradx) ]
		
		# Trust-region like constraint
		if trust_region:
			constraints.append( cp.norm(px) <= Delta)	

		# allow extra constraints (e.g., orthogonality for Grassmann manifold)
		constraints += search_constraints(x, px)

		# Append constraints from the domain
		constraints += domain._build_constraints(x + px)

		#with warnings.catch_warnings():
		#	warnings.simplefilter('ignore', PendingDeprecationWarning)

		try:
			problem = cp.Problem(cp.Minimize(pt), constraints)
			problem.solve(**kwargs)
		except cp.error.SolverError:
			break

		if problem.status in ['infeasible', 'unbounded']:
			raise Exception(problem.status)
	
		px = px.value
		pt = pt.value	
	
		if pt > 0:
			if verbose: print("No progress made on step")
			break

		# Backtracking line search
		alpha = 1. 
		stop = False
		for it2 in range(bt_maxiter):
			x_new = trajectory(x, px, alpha)
			fx_new = np.array(f(x_new))
			t_new = np.max(fx_new)

			# If x doesn't move enough, stop
			if np.max(np.abs(x_new - x)) < tol_dx:
				stop = True
				break

			if t_new <= t + c_armijo*alpha*pt:
				x = x_new
				fx = fx_new
				t = t_new
				Delta *= 2*alpha
				break

			# If predicted decrease is smaller than the tolerance, stop
			if np.abs(alpha*pt) < tol_df:
				stop = True
				break
			
			alpha = alpha*0.5

		if it2 == bt_maxiter-1:
			stop = True

		if verbose:
			print('%4d | %+14.10e | %8.2e | %8.2e |  %8.2e |' % (it+1, t, np.linalg.norm(px), alpha, Delta))

		if stop:
			break
	return x


