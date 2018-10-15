import numpy as np
from scipy.optimize import nnls
from linprog import linprog
from shared import LinProgException  

def minimax(f, x0, lb = None, ub = None, A_eq = None, b_eq = None, A_ub = None, b_ub = None,
		maxiter = 100, tol_dx = 1e-10, tol_df = 1e-10, tol_kkt = 1e-10, verbose = True,
		compute_kkt = True, use_lagrange = False, alpha0 = 1, c_armijo = 0.5,
		trajectory = None):
	""" Solves minimax optimization problems via sequential linearization and line search

		minimize_x  max_{i=1,ldots, M} f(x)
		such that   lb <= x <= ub
		            A_eq x = b_eq
                    A_ub x <= b_ub  

	This implements a minor modification of the algorithm described by Osborne and Watson [OW69]
	which consists of computing a descent direction via solving a linear program:

	(1)	minimize_{p_x, p_t}  p_t
		such that            t_k + p_t >= f(x_k) + p_x^T nabla f(x_k)

	and then performing a backtracking line search for the step length alpha:

	(2)	minimize_alpha  max_i f(x_k + alpha * p_x) <= t_k + alpha*p_t
	
	and these values updated:
		
	(3)	x_{k+1} = x_k + alpha * p_x
		t_{k+1} = t_k + alpha * p_t

	This second step (2) is the modification from [OW69] which used an exact line search.
	The Armijo like condition is added  

	References:
	[OW69]: Osborne and Watson, "An algorithm for minimax approximation in the nonlinear case",
		The Computer Journal, 12(1) 1969 pp. 63--68
		https://doi.org/10.1093/comjnl/12.1.63
	"""
	
	m = len(x0)
	maxiter_bt = 20
	# Check lb
	if lb is None:	lb = -np.inf*np.ones((m,))
	else: assert len(lb) == m, "lb has wrong dimension"
	
	# Check ub
	if ub is None: ub = np.inf*np.ones((m,))
	else: assert len(ub) == m, "ub has wrong dimension"

	# Check A_eq
	if A_eq is None:
		A_eq = np.zeros((0, m))
		b_eq = np.zeros(0)
	else:
		assert np.all(np.isclose(np.dot(A_eq,x0), b_eq)), "Initial x0 does not satisfy A_eq x0 = b_eq"

	# Check A_ub
	if A_ub is None:
		A_ub = np.zeros((0, m))
		b_ub = np.zeros(0)
	else:
		assert np.all( np.dot(A_ub, x0) <= b_ub + 1e-8), "Initial x0 does not satisfy A_ub x0 <= b_ub" 

	# Check trajectory
	if trajectory is None:
		trajectory = lambda x0, p, t: x0 + t * p

	# Create extened A_ub with bounds:
	A_ub_bnds = [A_ub]
	b_ub_bnds = [b_ub]
	for i in range(m):
		ei = np.zeros(m)
		ei[i] = 1.
		if np.isfinite(lb[i]):
			A_ub_bnds += [np.copy(ei).reshape(1,-1)]
			b_ub_bnds += [np.copy(lb[i])]
		if np.isfinite(ub[i]):
			A_ub_bnds += [np.copy(-ei).reshape(1,-1)]
			b_ub_bnds += [np.copy(-ub[i])]

	A_ub_bnds = np.vstack(A_ub_bnds)
	b_ub_bnds = np.hstack(b_ub_bnds)

	# Initialize
	fs = np.array([fi for fi in f(x0)])
	t = np.max(fs)
	t_new = t
	M = len(fs)

	x = np.copy(x0)

	# The equality constraints are the same at every step
	# so we compute these out here
	step_A_eq = np.hstack([A_eq, np.zeros((A_eq.shape[0],1)) ])
	step_b_eq = b_eq

	for it in range(maxiter):
		# Objective for the linear program
		# Variable order: x, t
		c = np.zeros((m + 1))
		c[m] = 1.
	
		step_lb = np.hstack([lb - x, -np.inf])
		step_ub = np.hstack([ub - x,  np.inf])
	
		# Inequality constraints coming from the domain
		step_A_ub1 = np.hstack([A_ub, np.zeros((A_ub.shape[0], 1))])
		step_b_ub1 = b_ub - np.dot(A_ub, x)

		# Inequality constraints from the linearization of the functions
		step_A_ub2 = np.zeros((M, m + 1 ))
		step_b_ub2 = np.zeros(M)
		
		for i, (fi, gi) in enumerate(f(x, return_gradient = True)):
			fs[i] = fi
			step_A_ub2[i,0:m] = gi 
			step_A_ub2[i,m] = -1
			step_b_ub2[i] = t - fi 
		
		if compute_kkt or use_lagrange: 	
			# Update Lagrange multipliers
			# Active constraints from the bounds
			active = (np.abs(np.max(fs) - fs) < 1e-7)
			A = [ step_A_ub2[active,:].T, ]
			
			# Active constraints from the inequalities on the domain
			active_ineq = (np.abs(np.dot(A_ub_bnds, x) - b_ub_bnds) < 1e-7)
			A += [ np.vstack([-A_ub_bnds[active_ineq].T, np.zeros((1,np.sum(active_ineq)))])]
			
			# Equality constraints are always active
			active_eq = np.ones(b_eq.shape[0], dtype = np.bool)
			# Since equality constraints have no sign constraint, add two copies to the non-negative LS problem
			# to mimic an unconstrained sign for these multipliers
			A += [ np.vstack([A_eq.T, np.zeros((1,A_eq.shape[0]))]), 
					np.vstack([-A_eq.T, np.zeros((1,A_eq.shape[0]))])]

			# Solve for the Langrange multipliers
			A = np.hstack(A)
			if A.shape[1] > 0:
				b = np.zeros(m+1)
				b[-1] = -1.
				lam2, kkt_res = nnls(A,b)
				if use_lagrange:
					# Update the objective using the Lagrange multipliers
					c -= np.dot(A, lam2)
			else:
				lam2 = np.zeros(A.shape[1])
			
			# Split multipliers back up for use later
			lam = np.zeros(M)
			lam[active] = lam2[:np.sum(active)]
			lam_ineq = np.zeros(A_ub_bnds.shape[0])
			lam_ineq[active_ineq] = lam2[np.sum(active):np.sum(active)+np.sum(active_ineq)]
			lam_eq = np.zeros(A_eq.shape[0])
			if A_eq.shape[0] > 0:
				lam_eq[active_eq] = lam2[-np.sum(active_eq):] 
			
		# Solve the linear program
		step_A_ub = np.vstack([step_A_ub1, step_A_ub2])
		step_b_ub = np.hstack([step_b_ub1, step_b_ub2])

		try:
			p = linprog(c, A_ub = step_A_ub, b_ub = step_b_ub,
						 A_eq = step_A_eq, b_eq = step_b_eq,
						 lb = step_lb, ub = step_ub)
		except LinProgException:
			#print step_A_ub
			#print step_b_ub
			#print step_A_eq
			#print step_b_eq
			if verbose: print "Linear program failed"
			break

		# Split solution up
		px = p[:m]
		pt = p[m]
		alpha = alpha0

		success = False
		for it2 in range(maxiter_bt):
			x_new = trajectory(x, px, alpha)
			if compute_kkt:
				# Update the KKT conditions
				kkt = np.zeros(m+1)
				kkt[-1] = 1
				kkt[:-1] -= np.dot(A_ub_bnds.T, lam_ineq)
				kkt[:-1] -= np.dot(A_eq.T, lam_eq)
				fs = np.zeros(M)
				try:
					for i, (fi, gi) in enumerate(f(x_new, return_gradient = True)):
						fs[i] = fi
						kkt[:-1] += lam[i]*gi
						kkt[-1] -= lam[i]
				except:
					break
			else:
				try:
					fs = np.array([fi for fi in f(x_new)])
				except:
					break
				kkt = np.nan

			t_new = np.max(fs)
				
			# Check the Armijo condition
			if t_new < t + c_armijo*pt*alpha:
				success = True
				break
			if np.linalg.norm(px) < tol_dx:
				break
			alpha *= 0.5

		kkt_norm = np.linalg.norm(kkt)
		move = np.linalg.norm(px*alpha)
		if not success: alpha = 0
		
		if verbose:
			if it == 0:
				print "iter |     objective     | bt step  |  norm dx |    kkt   |"
				print "   0 | %+14.10e |          |          |          |" % (t, )
			print "%4d | %+14.10e | %8.2e | %8.2e | %8.2e |" % (it+1, t_new, alpha, move, kkt_norm)
		if success:
			# Update points
			delta_f = t - t_new
			x = x_new
			t = t_new
			#t = np.mean([t + alpha*pt, t_new])
			#t += alpha*pt
		else:
			if verbose: print "No progress made on line search"
			break

		# Check early termination criteria
		if kkt_norm <= tol_kkt:
			if verbose: print "norm of KKT criteria less than tolerance"
			break
		if move <= tol_dx:
			if verbose: print "movement of x less than tolerance"
			break
		if delta_f <= tol_df:
			if verbose: print "change in objective function too small"
			break

	return x, t_new
