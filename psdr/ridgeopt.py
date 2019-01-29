""" Ridge-approximation based optimization"""


import numpy as np
import scipy.spatial.distance
import scipy.special
from function import Function
from polyridge import PolynomialRidgeApproximation 
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


	def _step_sequential(self, **kwargs):
		M, m = self.X.shape

		# Step 0: If not enough points have been sampled, sample those and stop
		if len(self.X) < M + 1:
			X_new = self.domain.sample(len(self.X) - (M + 1))
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

	
		idx = int(self.M_overdetermined + np.floor(self.it/2))
		
		if idx >= len(self.X):
			delta = np.inf
			domain = self.domain
		else:
			delta = np.sort(dist)[idx]
			domain = self.domain.add_constraints(ys = [xc], Ls = [np.eye(xc.shape[0])], rhos = [delta])
	

		# Step 3: Build ridge approximation for the objective function
		inside = domain.isinside(self.X)
		M_inside = np.sum(inside) 
		m = len(self.domain)
		if M_inside == len(self.domain) + 1:	
			# Linear case
			obj = PolynomialRidgeApproximation(subspace_dimension = 1, degree = 1)
		elif M_inside > scipy.special(m+2, 2):
			# Full quadratic case
			obj = PolynomialApproximation(degree = 2)
		else:
			# Otherwise we build a ridge approximation of an overdetermined dimension
			# Number of degrees of freedom in the sample set
			subspace_dim_candidates = np.arange(1, m+1)
			dof = np.array([ m*n + scipy.misc.comb(n + 2, 2) for n in subspace_dim_candidates])
			subspace_dimension = subspace_dim_candidates[np.max(np.argwhere(dof < M_inside).flatten())]
			obj = PolynomialRidgeApproximation(subspace_dimension = subspace_dimension, degree = 2)
		
		obj.fit(self.X[inside], self.fX[inside, 0])

		# Step 4: Build bounding linear surrogates for the constraints
		constraints = [PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'upper') 
			for i in range(1,len(self.fX[0]))
		]

		for i in range(len(self.fX[0])-1):
			constraints[i].fit(self.X[inside], self.fX[inside, i+1])

 

	def _step_old(self, **kwargs):
		r"""
		"""


		# Step 0: If not enough points have been sampled, sample those and stop
		if len(self.X) < self.M_overdetermined:
			X_new = self.domain.sample(self.M_overdetermined - len(self.X))
			fX_new = np.array([self.func(x) for x in X_new])
			self.X = np.vstack([self.X, X_new])
			if len(self.fX) == 0:
				self.fX = np.zeros((0, fX_new.shape[1])) 
			self.fX = np.vstack([self.fX, fX_new])
			return	

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

	
		idx = int(self.M_overdetermined + np.floor(self.it/2))
		if idx >= len(self.X):
			delta = np.inf
			domain = self.domain
		else:
			delta = np.sort(dist)[idx]
			domain = self.domain.add_constraints(ys = [xc], Ls = [np.eye(xc.shape[0])], rhos = [delta])
	
		# Step 3: Construct bounding approximations
		inside = domain.isinside(self.X)
	
		# TODO: Allow user specified surrogate classes	
		lb_surrogates = [ PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'lower')
			for i in range(len(self.fX[0]))]
		ub_surrogates = [ PolynomialRidgeApproximation(degree = 1, subspace_dimension = 1, bound = 'upper')
			for i in range(len(self.fX[0]))]

		for i, (lb_sur, ub_sur) in enumerate(zip(lb_surrogates, ub_surrogates)):
			# we also scale to [-1,1]
			Xin = self.X[inside]
			fXin = self.fX[inside,i]
			
			if i == 0:
				ub = np.max(fXin)
				lb = np.min(fXin)
				fXin = (fXin - lb)/(ub - lb)
			else:
				fXin = fXin/np.max(np.abs(fXin))
			
			lb_sur.fit(Xin, fXin)
			ub_sur.fit(Xin, fXin)

		# Step 4: Compute new solutions

		# Lower Bound Surrogate solution
		try:
			x_lb = sequential_lp(lb_surrogates[0], xc, lb_surrogates[0].grad, norm = None, 
					domain = domain,
					constraints = lb_surrogates[1:], 
					constraint_grads = [lb_sur.grad for lb_sur in lb_surrogates[1:]], 
					constraints_ub = np.zeros(len(lb_surrogates[1:])),
					**kwargs
				)
		except InfeasibleException:
			x_lb = sequential_lp(lb_surrogates[1:], xc, [lb_sur.grad for lb_sur in lb_surrogates[1:]],
				norm = 'hinge', domain = domain, 
				**kwargs
				)

		# Upper bound surrogate solution
		try:
			x_ub = sequential_lp(ub_surrogates[0], xc, ub_surrogates[0].grad, norm = None, 
					domain = domain,
					constraints = ub_surrogates[1:], 
					constraint_grads = [ub_sur.grad for ub_sur in ub_surrogates[1:]], 
					constraints_ub = np.zeros(len(ub_surrogates[1:])),
					**kwargs
				)
		except InfeasibleException:	
			x_ub = sequential_lp(ub_surrogates[1:], xc, [ub_sur.grad for ub_sur in ub_surrogates[1:]],
				norm = 'hinge', domain = domain,
				**kwargs
				)

		# TODO: Should we force x_lb and x_ub to be sufficiently distinct

		# Step 5: add new points to sample set
		fx_lb = self.func(x_lb)  	
		fx_ub = self.func(x_ub)  

		self.X = np.vstack([self.X, x_lb, x_ub])	
		self.fX = np.vstack([self.fX, fx_lb, fx_ub])	

		# To conclude: update iteration counter	
		self.it += 1


		if True:
			# Compute scores
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
				
			inside = domain.isinside(self.X) 
			sur_mismatch = np.linalg.norm(ub_surrogates[0](self.X[inside]) - lb_surrogates[0](self.X[inside]), 1)
		
			if self.it == 1:
				print "iter |     objective    |  infeasibility | # evals | TR Radius | Sur Mismatch |" 
				print "-----|------------------|----------------|---------|-----------|--------------|" 
			print "%4d | %16.9e | %14.7e | %7d | %9.2e | %12.3e |" % (self.it, objval, feasibility, 
				len(self.fX), delta, sur_mismatch)

			#print x_lb
			#print x_ub

	
if __name__ == '__main__':
	np.random.seed(1)

	from demos import GolinskiGearbox
	gb = GolinskiGearbox()

	X = gb.sample(10)
	fX = gb(X)

	opt = RidgeOptimization(gb, X = X, fX = fX)

	opt.step(100)
