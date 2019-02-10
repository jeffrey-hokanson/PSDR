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

	

	"""
	def __init__(self, objective, ineq_constraints, x0, domain = None):
		self.objective = objective
		self.ineq_constraints = ineq_constraints

		self.iter = 1	
		# Filter has pairs (constraint violation, objective value
		self.filter = [ (1e3,-np.inf)] 
		self.tr_radius = np.inf
		self.x = np.copy(x0)
		self.domain = domain

	def QP(self, x, tr_radius):
		r""" Solve the quadratic program subproblem
		"""
		# Step
		p = cp.Variable(x.shape[0])


		constraints = []

		# Trust region constraint
		if tr_radius < np.inf:
			constraints.append(cp.norm_inf(p) <= tr_radius)
		
		# Constraints from the domain
		if self.domain is not None:
			constraints.extend(self.domain._build_constraints(x + p))

		# Linearization of inequality constraints
		for con in self.ineq_constraints:
			# Linearization of constraint
			con_lin = con(x) + p.__rmatmul__(con.grad(x))
			constraints.append(con_lin <= 0)
			
		# Linearization of objective function
		obj = self.objective(x) + p.__rmatmul__(self.objective.grad(x))
		
		# Construct Hessian and remove negative definite directions
		H = self.objective.hessian(x)
		ew, ev = scipy.linalg.eigh(H)
		ew0 = np.maximum(ew, 0)
		H = ev.dot(np.diag(ew0)).dot(ev.T)

		obj += 0.5*cp.quad_form(p, H)

		prob = cp.Problem(cp.Minimize(obj), constraints)
		prob.solve()
		return p.value


	def filter_append(self, xk):
		r""" Append a point to the filter
		"""
		fk = float(self.objective(xk))
		hk = np.sum([max(con(xk),0) for con in self.ineq_constraints])

		# Parameters for filter acceptance
		gamma = 1e-5
		beta  = 1 - gamma

		# Remove dominated points from the filter
		for j, (hj, fj) in enumerate(self.filter):
			if hj <= beta*hk or fj + gamma*hj <= fk:
				self.filter.pop(j)

		# Append the new point to the filter
		self.filter.append( (hk,fk))
	 

	def filter_acceptible(self, xk):
		# Parameters for filter acceptance
		gamma = 1e-5
		beta  = 1 - gamma

		f = float(self.objective(xk))
		h = np.sum([max(con(xk),0) for con in self.ineq_constraints])
	
		for (hj, fj) in self.filter:
			if not (h <= beta*hj or f + gamma*h <= fj):
				return False

		return True 

	def restoration(self, xk):
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

		return p.value
		

	def step(self):
		
		xk = np.copy(self.x)
		try:
			p = self.QP(xk, self.tr_radius)
		except:
			self.filter_append(xk)
			p = self.restoration(xk)

		

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
	print sqp.restoration(x0)
	#sqp.step()
