""" Vector-valued output polynomial ridge approximation

"""
import numpy as np
import scipy.linalg
from polyrat import *
from .gn import gauss_newton 
from .initialization import initialize_subspace


def _grassmann_trajectory(U, Delta, t):
	# See HC18, eq. 23
	Y, s, ZT = scipy.linalg.svd(Delta, full_matrices = False, lapack_driver = 'gesvd')
	U_new = (U @ ZT.T) @ np.diag(np.cos(s*t)) + Y @ np.diag(np.sin(s*t))
	return U_new
	
def _gn_solver(J, r, subspace_dimension):
	Y, s, ZT = scipy.linalg.svd(J, full_matrices = False, lapack_driver = 'gesvd')
	# Apply the pseudoinverse
	n = subspace_dimension
	Delta = -ZT[:-n**2,:].T.dot(np.diag(1/s[:-n**2]).dot(Y[:,:-n**2].T.dot(r)))
	return Delta, s[:-n**2]
	

def _varpro_residual(U, X, fX, Basis, degree):
	# 
	Y = (U.T @ X.T).T
	basis = Basis(Y, degree)
	V = basis.basis()
	fX = fX.reshape(V.shape[0],-1)

	# Compute the linear coefficients
	if isinstance(basis, ArnoldiPolynomialBasis):
		c = V.T @ fX # np.array([V.T @ y for y in fX.T]).T
	else:
		#c = np.array([scipy.linalg.solve(V, y)[0].flatten() for y in fX.T])
		c = scipy.linalg.lstsq(V, fX)[0]

	r = fX - V @ c
	return r.T.flatten()

def _varpro_jacobian(U, X, fX, Basis, degree):
	M, m = X.shape
	m, n = U.shape
	
	fX = fX.reshape(M,-1)

	Y = (U.T @ X.T).T
	basis = Basis(Y, degree)

	V = basis.basis()
	DV = basis.vandermonde_derivative(Y)
	if isinstance(basis, ArnoldiPolynomialBasis):
		c = V.T @ fX
		Y = np.copy(V)
		s = np.ones(V.shape[1])
		ZT = np.eye(V.shape[1])
	else:
		Y, s, ZT = scipy.linalg.svd(V, full_matrices = False) 
		c = scipy.linalg.lstsq(V, fX)[0]
		
	r = fX - V @ c
		
	N = V.shape[1]

	Js = []	
	for ri, ci in zip(r.T, c.T):
		J1 = np.zeros((M,m,n))
		J2 = np.zeros((N,m,n))

		for ell in range(n):
			for k in range(m):
				DVDU_k = X[:,k,None]*DV[:,:,ell]
				# This is the first term in the VARPRO Jacobian minus the projector out fron
				J1[:, k, ell] = DVDU_k @ ci
				# This is the second term in the VARPRO Jacobian before applying V^-
				J2[:, k, ell] = DVDU_k.T @ ri

		# Project against the range of V
		J1 -= np.tensordot(Y, np.tensordot(Y.T, J1, (1,0)), (1,0))
		# Apply V^- by the pseudo inverse
		J2 = np.tensordot(np.diag(1./s),np.tensordot(ZT, J2, (1,0)), (1,0))
		J = -( J1 + np.tensordot(Y, J2, (1,0)))
		Js.append(J.reshape(J.shape[0], -1))
	
	return np.vstack(Js)


def polynomial_ridge_approximation(X, fX, dimension, degree, fixed_subspace = None, U0 = None, Basis = LegendrePolynomialBasis, **kwargs):
	r"""

	Parameters
	----------
	X: np.array (M, m)
		Input coordinates
	fX: np.array (M, ...)
		Outputs corresponding to input coordinates
	dimension: int
		The number of dimensions in the resulting ridge approximation
	degree: int
		The degree of the polynomial approximation
	fixed_subspace: None or np.array (m,n)
		
	U0: None or np.array(m, dimension)
		
	"""

	M, m = X.shape
	n = dimension
	fX = np.atleast_2d(fX)

	if U0 is None:
		A = np.hstack([initialize_subspace(X = X, fX = fXi) for fXi in fX.T]) 
		print(A)
		U0, _, _ = scipy.linalg.svd(A, full_matrices = False, compute_uv = True)
		U0 = U0[:, :dimension] 

	residual = lambda u: _varpro_residual(u.reshape(m,n), X, fX, Basis, degree)	
	jacobian = lambda u: _varpro_jacobian(u.reshape(m,n), X, fX, Basis, degree)	
	trajectory = lambda u, d, t: _grassmann_trajectory(u.reshape(m,n), d.reshape(m,n), t)
	gnsolver = lambda J, r: _gn_solver(J, r, n)

	u0 = U0.flatten()
	u, info = gauss_newton(residual, jacobian, u0,
		trajectory = trajectory, gnsolver = gnsolver, **kwargs) 

	U = u.reshape(m,n)
	return U

