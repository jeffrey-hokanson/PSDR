""" Ridge-approximation based optimization"""


import numpy as np
import scipy.spatial.distance
import scipy.special
import cvxpy as cp
from function import Function
from polyridge import PolynomialRidgeApproximation 
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

	References
	----------
	.. [BV04] Convex Optimization. Stephen Boyd and Lieven Vandenberghe. Cambridge University Press. 2004. 

	"""
	def __init__(self, func, X = None, fX = None):
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

		# Iteration counter	
		self.it = 0
	
		# number of points to be overdetermined:
		# TODO: Fix this computation
		self.M_overdetermined = len(self.domain)+2

	def step(self, maxiter = 1, **kwargs):

		for i in range(maxiter):
			self._step_sequential(**kwargs)
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
		print "%4d | %16.9e | %14.7e | %7d | %9.2e |" % (self.it, objval, feasibility, len(self.fX), self.delta )
		


	def _step_sequential(self, **kwargs):
		M, m = self.X.shape

		# Step 0: If not enough points have been sampled, sample those
		if len(self.X) < len(self.domain) + 1:
			X_new = self.domain.sample( m + 1 - len(self.X) ).reshape(-1,m)
			fX_new = np.array([self.func(x) for x in X_new])
			self.X = np.vstack([self.X, X_new])
			if len(self.fX) == 0:
				self.fX = np.zeros((0, fX_new.shape[1])) 
			self.fX = np.vstack([self.fX, fX_new])
			M, m = self.X.shape

		# Step 1: find center
		feasible = np.max(self.fX[:,1:], axis = 1) <= 0
		if np.sum(feasible) > 0:
			#print "feasible"
			# We have a feasible solution 
			score = np.copy(self.fX[:,0])
			score[~feasible] = np.inf
			j = np.argmin(score)
			xc = self.X[j]
		else:
			#print "infeasible"
			score = np.sum(np.maximum(self.fX[:,1:], 0), axis = 1)
			j = np.argmin(score)
			xc = self.X[j]

		
		# Step 2: Determine trust region radius
		dist = scipy.spatial.distance.cdist(xc.reshape(1,-1), self.X).flatten()
	
		idx = int( (len(self.domain) + 1) + np.floor(self.it/2))
		
		if idx >= len(self.X):
			self.delta = delta = np.inf
			domain = self.domain
		else:
			self.delta = delta = np.sort(dist)[idx]
			domain = self.domain.add_constraints(ys = [xc], Ls = [np.eye(xc.shape[0])], rhos = [delta])

		# Step 3: Build ridge approximation for the objective function
		inside = domain.isinside(self.X)
		M_inside = np.sum(inside) 
		if M_inside <= len(self.domain) + 3:	
			# Linear case
			obj = PolynomialRidgeApproximation(subspace_dimension = 1, degree = 1)
		elif M_inside > scipy.special.comb(m+2, 2):
			# Full quadratic case
			obj = PolynomialApproximation(degree = 2)
		else:
			# Otherwise we build a ridge approximation of an overdetermined dimension
			# Number of degrees of freedom in the sample set
			subspace_dim_candidates = np.arange(1, len(self.domain) + 1)
			dof = np.array([ len(self.domain)*n + scipy.misc.comb(n + 2, 2) for n in subspace_dim_candidates])
			subspace_dimension = subspace_dim_candidates[np.max(np.argwhere(dof < M_inside).flatten())]
			obj = PolynomialRidgeApproximation(subspace_dimension = subspace_dimension, degree = 2)
	
		# Step 3a: Scale to improve conditioning
		lb = np.min(self.fX[inside,0])	
		ub = np.max(self.fX[inside,0])
		if np.abs(ub - lb) < 1e-7: ub = lb+1e-7	
		obj.fit(self.X[inside], (self.fX[inside, 0] - lb)/(ub - lb))

		# Step 4: Build bounding linear surrogates for the constraints
		cons = [PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'upper') 
			for i in range(1,len(self.fX[0]))
		]
		for i in range(len(self.fX[0])-1):
			cons[i].fit(self.X[inside], self.fX[inside, i+1]/ np.max(np.abs(self.fX[inside,i+1])))

		# Step 5: Setup SQP Problem
		p = cp.Variable(len(self.domain))
		con_domain = domain._build_constraints(xc + p)
		try:
			# At first, we try to solve the SQP problem 
			H = obj.hessian(xc)
			ew = scipy.linalg.eigvalsh(H)
			# If indefinite, modify Hessian following Byrd, Schnabel, and Schultz
			# See: NW06 eq. 4.18
			if np.min(ew) < 0:
				H += np.abs(np.min(ew))*1.5*np.eye(H.shape[0])
				ew += 1.5*np.abs(np.min(ew))

			# (Convex) quadratic model of objective
			obj_quad = obj(xc) + p.__rmatmul__(obj.grad(xc)) 
			if np.min(ew) > 1e-10:
				obj_quad += cp.quad_form(p, H)
		
			cons_lin = []
			for con in cons:
				cons_lin.append( con(xc) + p.__rmatmul__(con.grad(xc)) <= 0)

			# Now solve QP
			problem = cp.Problem(cp.Minimize(obj_quad), cons_lin + con_domain)
			problem.solve(solver = 'ECOS')
			if problem.status in ['unbounded', 'error']:
				raise cp.SolverError

		except cp.SolverError:
			# If we can't solve the problem, solve the relaxed, ell-1 penality version
			# see, e.g., NW06 eq. 18.53
			# This uses a linear model of the objective 
			obj_pen = obj(xc) + p.__rmatmul__(obj.grad(xc))	
			mu = 10*np.linalg.norm(obj.grad(xc))
			for con in cons:
				# And addes a linearizd 
				obj_pen += cp.pos( con(xc) + p.__rmatmul__(con.grad(xc)) )

			problem = cp.Problem(cp.Minimize(obj_pen), con_domain)
			try:
				problem.solve(solver = 'ECOS')
			except cp.SolverError:
				print "failed"
				p.value = domain.sample() - xc

		x_new = xc + p.value
		# TODO: Check this point is sufficiently distinct
		dist = scipy.spatial.distance.cdist(x_new.reshape(1,-1), self.X).flatten()
		#print np.min(dist), np.max(dist)
		if np.min(dist) <= 1e-5*np.max(dist):
			x_new = domain.sample()

		fx_new = self.func(x_new)
		self.X = np.vstack([self.X, x_new])	
		self.fX = np.vstack([self.fX, fx_new])	

		# To conclude: update iteration counter	
		self.it += 1


	
if __name__ == '__main__':
	np.random.seed(1)

	from demos import GolinskiGearbox
	gb = GolinskiGearbox()

	#X = gb.sample()
	#fX = gb(X)

	#opt = RidgeOptimization(gb, X = X, fX = fX)
	opt = RidgeOptimization(gb)

	opt.step(500)
