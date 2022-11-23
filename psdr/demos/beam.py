from __future__ import print_function, division
import numpy as np
from psdr import Function, BoxDomain


BEAM = {
	'F': 5e3,		# Force applied to tip of beam
	'L': 1.5,		# Length of beam
	'E': 216.62e9, 	# Young's Modulus
	'nu': 0.27,
	'G': 86.65e9,	# Yeild Stress
}

class NowackiBeam(Function):
	r""" The Nowacki Beam Problem

	In this example, we seek to design a beam with fixed cross-sectional breath and height
	with respect to two objectives subject to five constraints described in [FSK08]_.

	The two parameters for this model and their ranges are given below.
	
    ====================================    ========================
    Variable                                Interpretation
    ====================================    ========================
    :math:`b \in [0.005, 0.050]`            breadth (m)
    :math:`h \in [0.020, 0.250]`            height (m)
    ====================================    ========================


	The following are the seven quantities of interest associated with this problem.
	
    ==========================    =================================================    ===============================================================
    Function name                 Formula                                              Role
    ==========================    =================================================    ===============================================================
    cross-sectional area          :math:`a = bh`                                       objective: minimize
    bending stress                :math:`\sigma = 6FL/(bh^2)`                          objective: minimize
    tip deflection                :math:`\delta = FL^3/(3EI_Y) \quad I_Y = bh^3/12`    constraint: :math:`\delta \le 0.005`
    bending stress                :math:`\sigma_B = 6FL/(bh^2)`                        constraint: :math:`\sigma_B \le \sigma_Y` 
    shear stress                  :math:`\tau = 3F/(2bh)`                              constraint: :math:`\tau \le \sigma_Y/2`
    height to breadth ratio       :math:`\rho = h/b`                                   constraint: :math:`\rho \le 10`
    tip force for buckling        :math:`F_T = (4/L^2)\sqrt{G I_T E I_Z/(1-\nu)}`      constraint: :math:`F_T \ge 2F`
    ==========================    =================================================    ===============================================================

	where :math:`I_T = (b^3h+bh^3)/12` and :math:`I_Z = b^3h/12`.
	The constants that appear above are taken from those values for mild steel:

	===============  =======================================
	Constant         Value
	===============  =======================================
	tip force        :math:`F=5\times 10^3` N
	beam length      :math:`L=1.5` m
	yield stress     :math:`\sigma_Y = 240\times 10^6` Pa
	Young's modulus  :math:`E=216.62\times 10^9` Pa
	Poisson's ratio  :math:`\nu=0.27`
	shear modulus    :math:`G=86.65\times 10^9` Pa
	===============  =======================================


	This example originates in [Now80]_.

	References
	----------
	.. [FSK08] Engineering Design via Surrogate Modeling: A Practical Guide
		Aleander I. J. Forrester, Andras Sobester, and Andy J. Keane
		Wiley, 2008
	.. [Now80] Modling of Design Decisions for CAD
		Horst Nowacki
		In: Computer Aided Design Modeling, J. Encarncao (editor)
		DOI:10.1007/BFb0040161
	"""

	def __init__(self):
		domain = build_beam_domain()
		funs = [
			beam_area, 
			beam_bending, 
			beam_tip_deflect, 
			beam_bending_stress,
			beam_shear_stress,
			beam_area_ratio,
			beam_twist_buckling,
		]
		grads = [
			beam_area_grad, 
			beam_bending_grad, 
			beam_tip_deflect_grad, 
			beam_bending_stress_grad,
			beam_shear_stress_grad,
			beam_area_ratio_grad,
			beam_twist_buckling_grad,
		]
		Function.__init__(self, funs, domain, grads = grads, vectorized = True, kwargs = BEAM)		

def build_beam_domain():
	# Note the book has an invalid range for height "(20mm > b > 250 mm)" and breadth "(10 mm > b > 50mm)"
	# Here we follow the dimensions implied in the corresponding matlab code
	# b = x(1)*0.045 +0.005
	# h = x(2)*0.23 + 0.02
	# I assume these definitions are in meters 
	domain = BoxDomain([5e-3, 0.02], [50e-3,0.25], names = ['breadth (m)', 'height (m)'])
	return domain

def beam_area(X, **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	area = X[:,0]*X[:,1]
	if len(X_shape) == 1:
		area = area[0]
	return area

def beam_area_grad(X, **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	grad = np.array([X[:,1], X[:,0]]).T
	if len(X_shape) == 1:
		grad = grad[0]
	return grad


def beam_bending(X, F = BEAM['F'], L = BEAM['L'], **kwargs):
	r""" Bending stress:

	
	"""
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]
	# From bending.m
	# SigmaB=(6*BeamProperties.F*BeamProperties.L)/(b*h^2);
	bending = 6*F*L/(b*h**2)
	if len(X_shape) == 1:
		bending = bending[0]
	return bending

def beam_bending_grad(X, F = BEAM['F'], L = BEAM['L'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]
	# From bending.m
	grad = np.array([-6*F*L/(b**2 * h**2), -2*6*F*L/(b * h**3)]).T
	if len(X_shape) == 1:
		grad = grad[0]
	return grad
	

def beam_tip_deflect(X, F = BEAM['F'], L = BEAM['L'], E = BEAM['E'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]
	
	Iy = b*h**3/12
	delta = F*L**3/(3*E*Iy)
	if len(X_shape) == 1:
		delta = delta[0]
	return delta

def beam_tip_deflect_grad(X, F = BEAM['F'], L = BEAM['L'], E = BEAM['E'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	grads = np.array([
		-4*F*L**3/(E*b**2*h**3),
		-12*F*L**3/(E*b*h**4)
	]).T
	if len(X_shape) == 1:
		grads = grads[0]
	return grads

def beam_bending_stress(X, F = BEAM['F'], L = BEAM['L'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]


	sigma_B = 6*F*L/(b*h**2)
	if len(X_shape) == 1:
		sigma_B = sigma_B[0]
	return sigma_B

def beam_bending_stress_grad(X, F = BEAM['F'], L = BEAM['L'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	grads = np.array([
		-6*F*L/(b**2*h**2),
		-12*F*L/(b*h**3),
	]).T	
	if len(X_shape) == 1:
		grads = grads[0]
	return grads


def beam_shear_stress(X, F = BEAM['F'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]


	sigma_Y = 3*F/(2*b*h)
	if len(X_shape) == 1:
		sigma_Y = sigma_Y[0]
	return sigma_Y

def beam_shear_stress_grad(X, F = BEAM['F'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	grads = np.array([
		-3*F/(2*b**2*h),
		-3*F/(2*b*h**2),
	]).T	
	if len(X_shape) == 1:
		grads = grads[0]
	return grads

def beam_area_ratio(X, **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	ratio = h/b
	if len(X_shape) == 1:
		ratio = ratio[0]
	return ratio

def beam_area_ratio_grad(X, **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	grads = np.array([
		-h/b**2, 
		1/b,
	]).T

	if len(X_shape) == 1:
		grads = grads[0]
	return grads


def beam_twist_buckling(X, F = BEAM['F'], L = BEAM['L'], G = BEAM['G'], E = BEAM['E'], nu = BEAM['nu'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	I_T = (b**3*h+b*h**3)/12
	I_Z = b**3*h/12

	twist = (4/L**2)*np.sqrt(G*I_T*E*I_Z/(1-nu))
	if len(X_shape) == 1:
		twist = twist[0]
	return twist

def beam_twist_buckling_grad(X, F = BEAM['F'], L = BEAM['L'], G = BEAM['G'], E = BEAM['E'], nu = BEAM['nu'], **kwargs):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]

	grads = np.array([
		2*np.sqrt(3)*np.sqrt(E*G*b**3*h*(b**3*h/12 + b*h**3/12)/(1 - nu))*(1 - nu)*(E*G*b**3*h*(b**2*h/4 + h**3/12)/(2*(1 - nu)) + 3*E*G*b**2*h*(b**3*h/12 + b*h**3/12)/(2*(1 - nu)))/(3*E*G*L**2*b**3*h*(b**3*h/12 + b*h**3/12)),
		2*np.sqrt(3)*np.sqrt(E*G*b**3*h*(b**3*h/12 + b*h**3/12)/(1 - nu))*(1 - nu)*(E*G*b**3*h*(b**3/12 + b*h**2/4)/(2*(1 - nu)) + E*G*b**3*(b**3*h/12 + b*h**3/12)/(2*(1 - nu)))/(3*E*G*L**2*b**3*h*(b**3*h/12 + b*h**3/12))
	]).T

	if len(X_shape) == 1:
		grads = grads[0]
	return grads
 
