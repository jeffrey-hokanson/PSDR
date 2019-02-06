import numpy as np
import scipy.linalg
import scipy.optimize
import cvxpy as cp


def sequential_qp(func, x0, cons, domain, maxiter = 100, verbose = True):
	r""" Solve a nonlinear optimization problem using a sequential-quadratic programming problem


	This implements the Bryd-Omojokun Trust Region SQP as described in Nocedal and Wright 06.

	.. math::

		\min{\mathbf{x} \in \mathcal{D} \subset \mathbb{R}^m} & \ f(\mathbf{x}) \\	
		\text{such that} & \ c_i(\mathbf{x}) \le 0

	"""

	x = np.copy(x0)

	tr_radius = np.inf
	mu = 1

	for it in range(maxiter):

		# Evaluate function/constraints
		fx = func(x)
		fgradx = func.grad(x)
		fHx = func.hessian(x)
		cx = [con(x) for con in cons]
		cgradx = [con.grad(x) for con in cons]
		
		# Compute "Lagrange" multipliers
		A = np.array(cgradx).T
		lam, res = scipy.optimize.nnls(-A, fgradx)
		if res < 1e-10 and np.linalg.norm(lam, np.inf) < 1e-10:
			break
		

		p = cp.Variable(x.shape[0])
		constraints = domain._build_constraints(x + p)

		obj = func(x) + p.__rmatmul__(func.grad(x))

		H = func.hessian(x)
		ew = scipy.linalg.eigvalsh(H)
		# If indefinite, modify Hessian following Byrd, Schnabel, and Schultz
		# See: NW06 eq. 4.18
		if np.min(ew) < 0:
			H += np.abs(np.min(ew))*1.5*np.eye(H.shape[0])
			ew += 1.5*np.abs(np.min(ew))
		
		obj += cp.quad_form(p, H)

		nonlinear_constraints = []
		for con in cons:
			con_lin = con(x) + p.__rmatmul__(con.grad(x))
			obj += mu*cp.pos( con_lin)
			nonlinear_constraints.append(con_lin <= con(x))

		prob = cp.Problem(cp.Minimize(obj), constraints + nonlinear_constraints)
		prob.solve()
		print p.value

		# Determine feasibility of iterate
		#cx = [con(x) for con in cons]
		#cx_grad = [con.grad(x) for con in cons]
		# Add linearized constraints
		#nonlin_constraints = [cxi + p.__rmatmul__(cx_gradi) for cxi, cx_gradi in zip(cx, cx_grad)]
		#print nonlin_constraints

		
		break

if __name__ == '__main__':
	np.random.seed(1)

	from demos import GolinskiGearbox
	from polyridge import PolynomialRidgeApproximation
	from poly import PolynomialApproximation
	gb = GolinskiGearbox()

	X = gb.sample(1000)
	#print X.shape
	fX = gb(X)
	
	obj = PolynomialApproximation(degree = 2)
	obj.fit(X, fX[:,0])
	con1 = PolynomialApproximation(degree = 2, bound = 'upper')
	con1.fit(X, fX[:,1])
	con2 = PolynomialApproximation(degree = 2, bound = 'upper')
	con2.fit(X, fX[:,2])

	x0 = -np.ones(len(gb.domain))
	sequential_qp(obj, x0, [con1, con2], gb.domain)




