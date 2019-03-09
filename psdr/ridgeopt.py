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
		\text{such that} & \ f_i(\mathbf{x}) \le 0.

	[This sign convention follows Ch. 5, [BV04]_ ]

	The approach taken here is to build a quadratic model of the objective function,
	using a quadratic ridge approximation if sufficient sample haven't been obtained,
	and build upper bounding linear surrogates for the constrainsts. 


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
		i.e., `constraints = lambda fx: fx[1:]`
.
	design_domain: Domain; optional
		If using chance or robust constraints, this partitions the input domain 

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


	@property
	def x(self):
		r""" The current best iterate
		"""
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

		return xc	

	@property
	def isfeasible(self):
		r""" Return true if the current best estimate is feasible; false otherwise.
		"""
		xc = self.x
		dist = scipy.spatial.distance.cdist(xc.reshape(1,-1), self.X).flatten()
		i = np.argmin(dist)
		constraint_vals = self.constraints(self.fX[i])
		if np.max(constraint_vals) <= 0:
			return True
		else:
			return False


	def _step(self, domain):
		
		# Find the current best iterate
		xc = self.x
		
		# Find the index of samples from which to build the surrogates
		I = self._surrogate_sample_index() 

		# Build the surrogate for the objective
		obj_vals = np.array([self.objective(fx) for fx in self.fX[I] ])
		obj_sur = self._build_objective_surrogate(self.X[I], obj_vals)

		# Build surrogates for the constraints 
		con_vals = np.array([self.constraints(fx) for fx in self.fX[I]])
		con_surs = [self._build_constraint_surrogate(self.X[I], con_val[:,j], j) for j in range(con_vals.shape[1])]
	
		# Solve SQP to find new sample
		x = np.copy(xc)
		for it2 in range(10):
			p = cp.Variables(x.shape[0])

			# Constraints from the domain
			constraints = domain._build_constraints(x + p)

			# Constraints from the nonlinear constraint functions
			for con_sur in con_surs:
				constraints.append( con_sur(x) + p.__rmatmul__(con_sur.grad(x)) <= 0)


			# Compute the Hessian of the objective function 
			H = obj_sur.hessian(x_new)
			ew = scipy.linalg.eigvalsh(H)
			
			# Linear model of objective
			obj = obj_sur(x) + p.__rmatmul__(obj_sur.grad(x))
			
			if np.max(np.abs(ew)) > 1e-10:
				if np.min(ew) < 0:
					# If indefinite, modify Hessian following Byrd, Schnabel, and Schultz
					# See: NW06 eq. 4.18
					H += np.abs(np.min(ew))*1.5*np.eye(H.shape[0])
				obj += cp.quad_form(p, H)
		
			# now solve QP
			problem = cp.problem(cp.minimize(obj_quad), linearized_constraints + constraints_domain)
			problem.solve()
			if problem.status in ['unbounded', 'error']:
				raise cp.solvererror


			# Update the point
			x += p.value

			# If surrogates are quadratic and linear, we don't need to keep solving SQP problem 
			if isinstance(obj_sur, (PolynomialApproximation, PolynomialRidgeApproximation)) and obj_sur.degree == 2:
				if all( [isinstance(con_sur, (PolynomialApproximation, PolynomialRidgeApproximation)) for con_sur in con_surs]) and \
					 all([con_sur.degree == 1 for con_sur in con_surs]):
					break 
	

	def _surrogate_sample_index(self):
		r""" Returns the index of those points to be used in constructing the surrogates

		"""		
		xc = self.xc
		# Find the distance of the best point to all others	
		dist = scipy.spatial.distance.cdist(xc.reshape(1,-1), self.X).flatten()

		# Number of points to contain
		M_contain = int( (len(self.domain) + 1) + np.floor(self.it/2))
	

		if M_contain >= len(self.X):
			return np.ones(self.X.shape[0], dtype = np.bool)
		else:
			# Sort ascending order
			tr_radius = np.sort(dist)[M_contain]
			return dist <= tr_radius

	def _build_objective_surrogate(self, X, fX):
		r""" Given a 
		"""			
		M, m = X.shape

		if M < m+1:
			raise UnderdeterminedException

		elif M < m + 3:
			# Linear surrogate
			obj_sur = PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1)

		elif M > scipy.special.comb(m+2,2):
			# Full quadratic surrogate
			obj_sur = PolynomialApproximation(degree = 2)

		else:
			subspace_dim_candidates = np.arange(1, m)
			dof = np.array([ m*n + scipy.misc.comb(n + 2, 2) for n in subspace_dim_candidates])
			subspace_dimension = subspace_dim_candidates[np.max(np.argwhere(dof < M).flatten())]
			obj_sur = PolynomialRidgeApproximation(subspace_dimension = subspace_dimension, degree = 2)
			
		# Fit the surrogate
		obj_sur.fit(X, fX)
		return obj_sur

	def _build_constraint_surrogate(self, X, fX, k):
		r"""
		"""
		con_sur = PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'upper')
		con_sur.fit(X, fX)
		return con_sur


	def _build_trust_region(self):
		pass	


	def step(self, maxiter = 1, **kwargs):

		for i in range(maxiter):
			self._find_center()
			self._find_trust_region()
			try:
				self._build_surrogates()
				x_new = self._optimize_surrogate()
				# If the new point is too close existing samples, sample randomly
				if len(self.X) > 0:
					dist = scipy.spatial.distance.cdist(x_new.reshape(1,-1), self.X).flatten()
					if np.min(dist) < 1e-5*min(self.tr_radius, 1):
						x_new = self._random_sample()

			except(UnderdeterminedException, cp.SolverError):
				x_new = self._random_sample()

				
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



	def _random_sample(self):
		# If we don't have enough points to build surrogates, sample randomly	
		if np.isfinite(self.tr_radius):
			domain = self.domain.add_constraints(ys = [self.xc], Ls = [np.eye(self.xc.shape[0])], rhos = [self.tr_radius])
		else:
			domain = self.domain
		x_new = domain.sample(1)
		# TODO: Use better sampling strategy
		return x_new



	def _find_trust_region(self):
		dist = scipy.spatial.distance.cdist(self.xc.reshape(1,-1), self.X).flatten()
		# Number of points to contain
		M_contain = int( (len(self.domain) + 1) + np.floor(self.it/2))
		
		if M_contain >= len(self.X):
			self.tr_radius = np.inf
		else:
			# Sort ascending order
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

			# now solve qp
			problem = cp.problem(cp.minimize(obj_quad), linearized_constraints + constraints_domain)
			problem.solve()
			if problem.status in ['unbounded', 'error']:
				raise cp.solvererror

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


class RidgeOptimizationUnderUncertainty(RidgeOptimization):
	r""" Ridge-based Optimization Under Uncertainty


	

	""" 
	def __init__(self, func, design_domain, random_domain,
		X = None, fX = None, objective = None, constraints = None):

		assert len(design_domain)+len(random_domain) == len(func.domain), (
			"The function provided has %d parameters but the design domain and random domains totaled %d" % (
				len(func.domain), len(design_domain)+ len(random_domain)))
		

		RidgeOptimization.__init__(self, func, X = X, fX = fX, objective = objective, constraints = constraints)

		# 


	
if __name__ == '__main__':
	np.random.seed(1)

	from demos import GolinskiGearbox
	gb = GolinskiGearbox()

	X = gb.sample(1000)
	#print X.shape
	fX = gb(X)
	#print fX.shape

	opt = RidgeOptimization(gb, X = X, fX = fX)
	print opt.x, opt.isfeasible
	#opt.step(200)
