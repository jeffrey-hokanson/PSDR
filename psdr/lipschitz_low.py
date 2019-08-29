from __future__ import division, print_function

import numpy as np
import cvxpy as cp

from .lipschitz import LipschitzMatrix

from .domains import DEFAULT_CVXPY_KWARGS
from .misc import merge
from .opg import opg_grads

class LowRankLipschitzMatrix(LipschitzMatrix):

	def __init__(self, rank, epsilon = 0, verbose = True, U0 = None, maxiter = 50, 
		tol_obj = 1e-9, tol_dx = 1e-7, solver_kwargs = {}):
		self.rank = rank
		self.verbose = verbose
		if U0 is not None:
			self.U0 = np.atleast_2d(np.array(U0))
		else:
			self.U0 = None

		self.maxiter = maxiter
		self.tol_obj = tol_obj
		self.tol_dx = tol_dx
		self.solver_kwargs = merge(DEFAULT_CVXPY_KWARGS, solver_kwargs)
		self.epsilon = epsilon

	def _fit(self, X, fX, grads, epsilon, scale):
		m = grads.shape[1]

		if self.rank == m:
			U = np.eye(m)
			J, alpha = self._fixed_U(U, X, fX/scale, grads/scale, epsilon)
		elif self.rank < m:		
			if self.U0 is None:
				U0 = self._init_U(X, fX, grads)
			else:
				# Checks on dimension of U0
				assert self.U0.shape[0] == X.shape[1]
				assert self.U0.shape[1] == self.rank
				U0 = self.U0
		
			U, J, alpha = self._optimize(U0, X, fX/scale, grads/scale, epsilon)
		else:
			raise ValueError("Rank must be less than or equal to the dimension of the space")

		# Rotate U
		ew, ev = np.linalg.eigh(J)
		# Flip to descending order
		ew = ew[::-1]
		ev = ev[:,::-1]
		J = np.diag(ew)
		U = U.dot(ev)
		# NB: since J is symmetric, its eigenvectors are orthogonal and hence
		# after the rotation below, U is still unitary
		#U = U.dot(ev[:,::-1])

		# TODO: sign flipping

		# Fix scaling	
		J *= scale**2
		alpha *= scale**2

		self._U = U
		self._J = J
		self._alpha = alpha

	@property
	def H(self):
		U = self.U
		J = self.J
		m, r = U.shape
		#Q, _ = np.linalg.qr(self.U, mode = 'complete')
		#JJ = np.diag(np.hstack([np.diag(self.J), self.alpha*np.ones(m-r)]))
		#return Q.T.dot(JJ.dot(Q))
		return U.dot(J).dot(U.T) + self.alpha*(np.eye(m) - U.dot(U.T))

	@property
	def L(self):
		m, r = self.U.shape
		Q, _ = np.linalg.qr(self.U, mode = 'complete')
		JJdiag = np.hstack([np.diag(self.J), self.alpha*np.ones(m-r)])
		return Q.dot(np.diag(np.sqrt(JJdiag))).dot(Q.T)

	@property
	def J(self):
		return self._J

	@property
	def U(self):
		return self._U

	@property
	def alpha(self):
		return self._alpha

	def _init_U(self, X, fX, grads):
		if len(X) > X.shape[1]:
			all_grads = np.vstack([grads, opg_grads(X, fX)])
		else:
			all_grads = grads	
		assert len(all_grads) >= self.rank, "Not enough samples or gradients provided to estimate an initial subspace"
	
		U, s, VT = np.linalg.svd(all_grads.T, full_matrices = False)
		return U[:,0:self.rank]

	
	def _fixed_U(self, U, X, fX, grads, epsilon):
		r"""Compute J and alpha for a fixed U

		This uses a straightfoward CVXPY implementation

		Returns
		-------
		J: np.array (r, r)
			Low rank block
		alpha: float
			scalar multiple for the remainder of the matrix
		"""	

		m, r = U.shape
		J = cp.Variable((r,r), PSD = True)
		alpha = cp.Variable(nonneg = True)

		obj = cp.trace(J) + alpha*(m-r)

		constraints = []
		if m == r:
			constraints.append(alpha == 0)

		for i in range(len(X)):
			for j in range(i+1, len(X)):
				lhs = (np.abs(fX[i] - fX[j]) - epsilon)**2
				y = X[i] - X[j]
				UTy = U.T.dot(y)
				rhs = cp.quad_form(UTy, J) + alpha*( y.dot(y) - UTy.dot(UTy))
				constraints.append(lhs <= rhs)

		
		Pperp = np.eye(m) - U.dot(U.T) 
		for g in grads:
			lhs = np.outer(g,g)
			#rhs = cp.quad_form(U, J) + alpha*Pperp
			rhs = (J.__rmatmul__(U)).__matmul__(U.T) + alpha*Pperp
			constraints.append(lhs << rhs)

		problem = cp.Problem(cp.Minimize(obj), constraints)
		problem.solve(**self.solver_kwargs)

		J = np.array(J.value)
		alpha = float(alpha.value)
		return J, alpha

	def _U_descent(self, U, J, alpha, X, fX, grads, epsilon):
		r""" compute a descent direction for U via a linearization of the problem 
		""" 

		m, r = U.shape

		Jp = cp.Variable((r,r), symmetric = True)
		alphap = cp.Variable()
		Up = cp.Variable((m,r))		

		obj = cp.trace(Jp) + alphap*(m - r)
		
		# Orthogonality constraint from manifold considerations
		constraints = [ Up.__rmatmul__(U.T) == 0]
		# Also require that the new step could not take the problem negative
		#constraints.append(alphap >= -alpha)
		
		for i in range(len(X)):	
			for j in range(i+1, len(X)):
				y = X[i] - X[j]
				UTy = U.T.dot(y)
				lhs = (np.abs(fX[i] - fX[j]) - epsilon)**2 
				# y.T @ U.T @ J @  U @ y + alpha* y.T @ (I - U @ U.T) @ y
				lhs -= UTy.dot(J.dot(UTy)) + alpha*(y.dot(y) - UTy.dot(UTy)) # 0th order term
				# 2 * y.T @ U.T @ J @ Up @ y + y.T @ U.T @ Jp @ U @ y
				rhs = (2*Up.T.__rmatmul__(J.dot(UTy))).__matmul__(y) + cp.quad_form(UTy, Jp) # 1st order term
				#rhs = 2*Up.T.__matmul__(y).__rmatmul__(J.dot(UTy)) + cp.quad_form(UTy, Jp)
				rhs += -2*alpha*(Up.T.__matmul__(y)).__rmatmul__(J.dot(UTy).T) + alphap*(y.dot(y) - UTy.dot(UTy))
				constraints.append(lhs <= rhs)

		Pperp = np.eye(m) - U.dot(U.T) 
		for g in grads:
			lhs = np.outer(g, g)
			lhs -= U.dot(J.dot(U.T)) + alpha*Pperp # 0th order term
	
			UpJUT = Up.__matmul__(J.dot(U.T))
			UpUT = Up.__matmul__(U.T)
			rhs = (Jp.__rmatmul__(U)).__matmul__(U.T) + UpJUT + UpJUT.T
			rhs -= alpha*(UpUT + UpUT.T)
			rhs += alphap*Pperp
		
			constraints.append(lhs << rhs) 

		problem = cp.Problem(cp.Minimize(obj), constraints)
		problem.solve(**self.solver_kwargs)	
		if problem.status not in ['optimal']:
			raise ValueError("Could not find search direction: exited with %s" % problem.status) 
		Up = np.array(Up.value)
		Jp = np.array(Jp.value)
		alphap = float(alphap.value)

		return Up, Jp, alphap	

	def _optimize(self, U0, X, fX, grads, epsilon):
		U = np.copy(U0)
		m, r = U.shape

		J, alpha = self._fixed_U(U, X, fX, grads, epsilon) 
		obj = np.trace(J) + (m - r)*alpha

		if self.verbose:
			lam = np.linalg.eigvalsh(J)
			# Header
			print("iter |     objective    |  norm dU | step len |  lam_max |  lam_min |   alpha  |")
			print("-----|------------------|----------|----------|----------|----------|----------|")
			print("%4d | %14.10e |          |          | %7.2e | %7.2e | %7.2e |" % (0, obj, np.max(lam), np.min(lam), alpha))	

		for it in range(self.maxiter):
			dU, dJ, dalpha = self._U_descent(U, J, alpha, X, fX, grads, epsilon)

			if np.linalg.norm(dU) < self.tol_dx:
				if self.verbose:
					print("stopped due to small norm dU")
				break

			t = 1
			Y, s, ZT = np.linalg.svd(dU, full_matrices = False)
			# As each backtracking step requires solving a semidefinite program
			# we elect to aggressively make the step size smaller to reduce the number of 
			# backtracking steps
			for it2 in range(10):
				U_new = U.dot(ZT.T.dot(np.diag(np.cos(s*t)).dot(ZT))) + Y.dot(np.diag(np.sin(s*t)).dot(ZT))
				J_new, alpha_new = self._fixed_U(U_new, X, fX, grads, epsilon)
				obj_new = np.trace(J_new) + (m - r)*alpha_new
				if obj_new < obj:
					break
				t *= 0.1
			
			if obj_new > obj:
				t = 0
			
			if self.verbose:
				lam = np.linalg.eigvalsh(J_new)
				print("%4d | %14.10e | %7.2e | %7.2e | %7.2e | %7.2e | %7.2e |" % (it+1, obj_new, np.linalg.norm(dU), t, np.max(lam), np.min(lam), alpha_new))
			
			if obj_new > obj:
				if self.verbose:
					print("objective did not decrease during line search")
				break
			
			# Rotate
			#ew, ev = np.linalg.eigh(J)
			#U = U_new.dot(ev.T)
			#J = np.diag(ew)
			U = U_new
			J = J_new
			alpha = alpha_new

			if obj_new + self.tol_obj > obj:
				if self.verbose:
					print("stopped due to small change in objective function")
				break
			obj = obj_new

		return U, J, alpha

	
