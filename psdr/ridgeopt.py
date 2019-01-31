""" Ridge-approximation based optimization"""


import numpy as np
import scipy.spatial.distance
import scipy.special
import cvxpy as cp
from function import Function
from polyridge import PolynomialRidgeApproximation, UnderdeterminedException 
from poly import PolynomialApproximation
from seqlp import sequential_lp, InfeasibleException

class RidgeOptimization:
	r""" Ridge-based nonlinear optimization

	Given a vector valued function :math:`\mathbf{f}: \mathcal{D}\subset \mathbb{R}^m\to \mathbb{R}^n`,
	this class solves the optimization problem

	.. math::

		\min_{\mathbf{x}\in \mathcal{D}} &\ f_0(\mathbf{x}) \\
		\text{such that} & \ f_i(\mathbf{x}) \le 0

	by constructing a sequence of bounding surrogates.
	[This sign convention follows Ch. 5, [BV04]_ ]


	Parameters
	----------
	func : Function
		Function to optimize
	X: array-like (M,m); optional
		Points at which the function has been evaluated
	fX: array-like (M,n); optional 
		Vector-valued output of the function
	objective: callable; optional
		A function that takes the output `fx=func(x)`
		and generates the scalar valued objective function.
		This defaults to the first output;
		i.e., `objective = lambda fx: fx[0]`.
	constraints: callable; optional
		A function that takes the output `fx=func(x)`
		and generates a vector-valued constraint function;
		by default this yields all but the first outputs;
		i.e., `constraints = lambda fx: fx[1:]`.

	References
	----------
	.. [BV04] Convex Optimization. Stephen Boyd and Lieven Vandenberghe. Cambridge University Press. 2004. 

	"""
	def __init__(self, func, X = None, fX = None, 
		objective = None, constraints = None):
		
		# Iteration counter	
		self.it = 0

		self.func = func
		self.domain = func.domain_norm 
		

		# Convert input into list
		if X is None:
			self.X = np.zeros((0,len(self.domain)))
		else:
			self.X = np.array(X)

		if fX is None:
			self.fX = np.zeros((0,))
		else:
			self.fX = np.array(fX)

		if objective is None:
			objective = lambda fx: fx[0]

		self.objective = objective

		if constraints is None:
			constraints = lambda fx: fx[1:]

		self.constraints = constraints	

	def step(self, maxiter = 1, **kwargs):

		for i in range(maxiter):
			self._find_center()
			self._find_trust_region()
			try:
				self._build_surrogates()
				x_new = self._optimize_surrogate()
			except(UnderdeterminedException, cp.SolverError):
				x_new = self._underdetermined_sample()

			fx_new = self.func(x_new)
			self.X = np.vstack([self.X, x_new])	
			if len(self.fX) == 0:
				self.fX = fx_new.reshape(1,-1)
			else:
				self.fX = np.vstack([self.fX, fx_new])	
			
			self.it += 1 
			self._print()
	


	def _print(self):
		feasible = np.max(self.fX[:,1:], axis = 1) <= 0
		if np.sum(feasible) > 0:
			score = np.copy(self.fX[:,0])
			score[~feasible] = np.inf
			j = np.argmin(score)
			objval = self.fX[j,0]
			feasibility = np.sum(np.maximum(self.fX[j,1:],0))
		else:
			objval = np.nan
			score = np.sum(np.maximum(self.fX[:,1:],0), axis = 1)
			feasibility = np.min(score)
			objval = np.nan
		
		if self.it == 1:
			print "iter |     objective    |  infeasibility | # evals | TR Radius |" 
			print "-----|------------------|----------------|---------|-----------|" 
		print "%4d | %16.9e | %14.7e | %7d | %9.2e |" % (self.it, objval, feasibility, len(self.fX), self.tr_radius )



	def _underdetermined_sample(self):
		# If we don't have enough points to build surrogates, sample randomly		
		x_new = self.domain.sample(1)
		# TODO: Use better sampling strategy
		return x_new


	def _find_center(self):
		obj_vals = np.array([self.objective(fx) for fx in self.fX])
		constraint_vals = np.array([self.constraints(fx) for fx in self.fX])
		
		if len(obj_vals) == 0:
			self.xc = xc = self.domain.sample()
			return xc

		feasible = np.max(constraint_vals, axis = 1) <= 0
		if np.sum(feasible) > 0:
			# There is at least one feasible point
			score = np.copy(obj_vals)
			score[~feasible] = np.inf
			j = np.argmin(score)
			xc = self.X[j]
		else:
			# Otherwise we choose the function
			score = np.sum(np.maximum(constraint_vals, 0), axis = 1)
			j = np.argmin(score)
			xc = self.X[j]

		self.xc = xc
		return xc	


	def _find_trust_region(self):
		dist = scipy.spatial.distance.cdist(self.xc.reshape(1,-1), self.X).flatten()
		
		# Number of points to contain
		M_contain = int( (len(self.domain) + 1) + np.floor(self.it/2))
		
		if M_contain >= len(self.X):
			self.tr_radius = np.inf
		else:
			self.tr_radius = np.sort(dist)[M_contain]

		return self.tr_radius


	def _build_surrogates(self, **kwargs):
		objective_vals = np.array([self.objective(fx) for fx in self.fX])
		constraint_vals = np.array([self.constraints(fx) for fx in self.fX])
		
		# Determine the points that are inside
		if np.isfinite(self.tr_radius):
			domain = self.domain.add_constraints(ys = [self.xc], Ls = [np.eye(self.xc.shape[0])], rhos = [self.tr_radius])
		else:
			domain = self.domain

		inside = domain.isinside(self.X)
		M_inside = np.sum(inside) 

		#########################################################################################
		# Build the surrogate for the objective function
		#########################################################################################
		if M_inside <= len(domain) + 1:
			raise UnderdeterminedException

		if M_inside <= len(domain) + 3:	
			# Linear case
			self.objective_surrogate = PolynomialRidgeApproximation(subspace_dimension = 1, degree = 1)

		elif M_inside > scipy.special.comb(len(domain)+2, 2):
			# Full quadratic case
			self.objective_surrogate = PolynomialApproximation(degree = 2)

		else:
			# Otherwise we build a ridge approximation of an overdetermined dimension
			# Number of degrees of freedom in the sample set
			subspace_dim_candidates = np.arange(1, len(self.domain) + 1)
			dof = np.array([ len(self.domain)*n + scipy.misc.comb(n + 2, 2) for n in subspace_dim_candidates])
			subspace_dimension = subspace_dim_candidates[np.max(np.argwhere(dof < M_inside).flatten())]
			self.objective_surrogate = PolynomialRidgeApproximation(subspace_dimension = subspace_dimension, degree = 2)

		# Scale to improve conditioning
		lb = np.min(objective_vals)	
		ub = np.max(objective_vals)
		if np.abs(ub - lb) < 1e-7: ub = lb+1e-7	
		self.objective_surrogate.fit(self.X[inside], (objective_vals[inside] - lb)/(ub - lb))
		
		#########################################################################################
		# Build the surrogates for the constraints
		#########################################################################################
		self.constraint_surrogates = [PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'upper') 
			for i in range(len(constraint_vals[0]))]

		for i in range(len(constraint_vals[0])):
			self.constraint_surrogates[i].fit(self.X[inside], constraint_vals[inside, i]/ np.max(np.abs(constraint_vals[inside, i])))
			
	def _optimize_surrogate(self):

		# Setup the domain
		if np.isfinite(self.tr_radius):
			domain = self.domain.add_constraints(ys = [self.xc], Ls = [np.eye(self.xc.shape[0])], rhos = [self.tr_radius])
		else:
			domain = self.domain

		p = cp.Variable(len(domain))
		
		constraints_domain = domain._build_constraints(self.xc + p)
		
		try:
			# At first, we try to solve the SQP problem 
			H = self.objective_surrogate.hessian(self.xc)
			ew = scipy.linalg.eigvalsh(H)
			# If indefinite, modify Hessian following Byrd, Schnabel, and Schultz
			# See: NW06 eq. 4.18
			if np.min(ew) < 0:
				H += np.abs(np.min(ew))*1.5*np.eye(H.shape[0])
				ew += 1.5*np.abs(np.min(ew))

			# Linear model of objective
			obj_quad = self.objective_surrogate(self.xc)
			obj_quad += p.__rmatmul__(self.objective_surrogate.grad(self.xc)) 
			if np.min(ew) > 1e-10:
				# Add quadratic term if non-zero
				obj_quad += cp.quad_form(p, H)
	
			# Linearize the constraints	
			linearized_constraints = []
			for con in self.constraint_surrogates:
				linearized_constraints.append( con(self.xc) + p.__rmatmul__(con.grad(self.xc)) <= 0)

			# Now solve QP
			problem = cp.Problem(cp.Minimize(obj_quad), linearized_constraints + constraints_domain)
			problem.solve()
			if problem.status in ['unbounded', 'error']:
				raise cp.SolverError

		except cp.SolverError:
			# If we can't solve the problem, solve the relaxed, ell-1 penality version
			# see, e.g., NW06 eq. 18.53
			# This uses a linear model of the objective 
			obj_pen = self.objective_surrogate(self.xc) + p.__rmatmul__(self.objective_surrogate.grad(self.xc))	
			mu = 10*np.linalg.norm(self.objective_surrogate.grad(self.xc))
			for con in self.constraint_surrogates:
				# And addes a linearizd 
				obj_pen += cp.pos( con(self.xc) + p.__rmatmul__(con.grad(self.xc)) )

			problem = cp.Problem(cp.Minimize(obj_pen), constraints_domain)
			problem.solve()

		x_new = self.xc + p.value
		return x_new


	
if __name__ == '__main__':
	np.random.seed(1)

	from demos import GolinskiGearbox
	gb = GolinskiGearbox()

	#X = gb.sample(100)
	#print X.shape
	#fX = gb(X)
	#print fX.shape

	opt = RidgeOptimization(gb)
	opt.step(50)
