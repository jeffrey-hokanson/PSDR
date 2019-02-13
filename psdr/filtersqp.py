import numpy as np
import scipy.linalg
import cvxpy as cp
import sys

# TODO: wrap evaluation/gradient/hessian in objective and constraints
#if sys.version_info[0] < 3:
#	from functools32 import lru_cache
#else:
#	from functools import lru_cache



class FilterSQP(object):
	r""" A filter based sequential quadratic program implementation

	This class solves the nonlinear programming problem

	.. math::
		
		\min{\mathbf{x} \in \mathcal{D} \subset \mathbb{R}^m} & \ f(\mathbf{x}) \\	
		\text{such that} & \ c_i(\mathbf{x}) \le 0

	using a trust-region filter-SQP method following Fletcher, Leyffer, and Toint [FLT02]_.

	
	References
	----------
	.. [FLT02] Roger Fletcher, Sven Leyffer, and Philippe L. Toint.
		On the Global Convergence of a Filter-SQP Algorithm.
		SIAM J. Optim. Vol 13, No. 1. pp 44-59.	

	"""
	def __init__(self, objective, ineq_constraints, x0, domain = None):
		self.objective = objective
		self.ineq_constraints = ineq_constraints

		self.iter = 1	
		# Filter has pairs (constraint violation, objective value)
		self.filter = [ (1e3,-np.inf)] 
		self.tr_radius = np.inf
		self.x = np.copy(x0)
		self.domain = domain

		self.rho0 = 1.		# \rho^\circ in FLT02
		self.xtol = 1e-10

		# Parameters for filter acceptability
		self.gamma = 1e-5
		self.beta = 1 - self.gamma

		# 
		self.sigma = 1e-4

		self.done = False

	def QP(self, x, rho):
		r""" Solve the quadratic program subproblem
		"""
		# Step
		p = cp.Variable(x.shape[0])

		constraints = []

		# Trust region constraint
		if rho < np.inf:
			constraints.append(cp.norm_inf(p) <= rho)
		
		# Constraints from the domain
		if self.domain is not None:
			constraints.extend(self.domain._build_constraints(x + p))

		# Linearization of inequality constraints
		for con in self.ineq_constraints:
			# Linearization of constraint
			con_lin = con(x) + p.__rmatmul__(con.grad(x))
			constraints.append(con_lin <= 0)
			
		# Linearization of objective function
		#obj = self.objective(x) + p.__rmatmul__(self.objective.grad(x))
		obj = p.__rmatmul__(self.objective.grad(x))
		
		# Construct Hessian and remove negative definite directions so we maintain a convex QP problem
		H = self.objective.hessian(x)
		ew, ev = scipy.linalg.eigh(H)
		ew0 = np.maximum(ew, 0)
		H = ev.dot(np.diag(ew0)).dot(ev.T)

		obj += 0.5*cp.quad_form(p, H)

		# Solve the QP for the step
		prob = cp.Problem(cp.Minimize(obj), constraints)
		prob.solve()

		#print "Objective value", obj.value
		#print "Objective value", -p.value.dot(self.objective.grad(x)) - 0.5*p.value.T.dot(H.dot(p.value))

		return p.value, -obj.value


	def filter_append(self, xk):
		r""" Append a point to the filter
		"""
		fk = float(self.objective(xk))
		hk = np.sum([max(con(xk),0) for con in self.ineq_constraints])

		# Remove dominated points from the filter
		for j, (hj, fj) in enumerate(self.filter):
			if hj <= self.beta*hk or fj + self.gamma*hj <= fk:
				self.filter.pop(j)

		# Append the new point to the filter
		self.filter.append( (hk,fk))
	 

	def filter_acceptible(self, xnew, xk = None):

		f = float(self.objective(xnew))
		h = np.sum([max(con(xnew),0) for con in self.ineq_constraints])
	
		filter_ = self.filter[:]
 
		if xk is not None:
			fk = float(self.objective(xk))
			hk = np.sum([max(con(xk),0) for con in self.ineq_constraints])
			filter_.append( (hk, fk) )				
		
		for (hj, fj) in self.filter:
			if not (h <= self.beta*hj or f + self.gamma*h <= fj):
				return False

		return True 

	def restoration(self, xk, rho):
		r""" Decrease constraint violation in an attempt to find compatible constraints
		"""

		p = cp.Variable(len(xk))

		constraints = []

		# Trust region constraint
		if self.tr_radius < np.inf:
			constraints.append(cp.norm_inf(p) <= tr_radius)
		
		# Constraints from the domain
		if self.domain is not None:
			constraints.extend(self.domain._build_constraints(xk + p))
				
		# setup objective as linearization of the constraints
		obj = 0
		for con in self.ineq_constraints:
			obj += cp.pos(con(xk) + p.__rmatmul__(con.grad(xk)))
			#obj += cp.scalene(con(xk) + p.__rmatmul__(con.grad(xk)), 1, 1e-4)

		prob = cp.Problem(cp.Minimize(obj), constraints)
		prob.solve()

		return p.value, -obj.value
	

	def solve(self, maxiter = 100):
		for it in range(maxiter):
			if self.done:
				break
			self.step()
	

	def step(self):
		
		xk = np.copy(self.x)

		rho = 1.*self.rho0

		try:
			for it in range(20):
				p, pred = self.QP(xk, rho)

				# If the step is too small, stop 
				if np.linalg.norm(p) < self.xtol:
					print "Done: Satisfies KKT conditions"
					self.done = True
					return
				
				# actual reduction
				ared = self.objective(xk) - self.objective(xk+p)

				# Is the new candidate point acceptible to the filter
				if self.filter_acceptible(xk + p, xk):
					if not ared < self.sigma * pred:
						print "step passes filter but insuffient decrease of objective"
					elif pred > 0:
						print "step passes filter but predicted objective increases"
					else:
						print "step accepted"
						if pred < 0:
							self.filter_append(xk)
						self.x = xk + p
						self.iter += 1
						return
				else:
					print "Step fails filter"	
				rho *= 0.5
		except:
			# If we are in the restoration phase (i.e., we cannot solve the QP)
			# we try to minimize the constraint violation until we find a point
			# that is (1) acceptible to the filter and (2) for which we have a compatible QP.

			rho = 1.*self.rho0
			self.filter_append(xk)
			
			# we use a trust-region approach to ensure we get a feasible decrease
			for it in range(20):
				p, pred = self.restoration(xk, rho)

				ared = np.sum([max(con(xk), 0) - max(con(xk+p),0) for con in self.ineq_constraints])
			
				if self.filter_acceptible(xk + p):
					if ared < self.sigma*pred and pred < 0:
						self.x = xk + p
						self.iter += 1
						return
		

if __name__ == '__main__':
	np.random.seed(1)

	from demos import GolinskiGearbox
	from polyridge import PolynomialRidgeApproximation
	from poly import PolynomialApproximation
	gb = GolinskiGearbox()

	X = gb.sample(1000)
	fX = gb(X)

	# Build objective and surrogates	
	obj = PolynomialApproximation(degree = 2)
	obj.fit(X, fX[:,0])
	con1 = PolynomialApproximation(degree = 2, bound = 'upper')
	con1.fit(X, fX[:,1])
	con2 = PolynomialApproximation(degree = 2, bound = 'upper')
	con2.fit(X, fX[:,2])

	x0 = -1*np.ones(len(gb.domain))
	print gb.domain.lb

	sqp = FilterSQP(obj, [con1, con2], x0, gb.domain)
	#sqp.filter_append(x0)
	#print sqp.QP(x0, 1)

	for it in range(10):
		sqp.step()
		print sqp.x
	
