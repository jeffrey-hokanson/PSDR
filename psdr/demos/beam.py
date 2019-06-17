from __future__ import print_function, division
import numpy as np
from psdr import Function, BoxDomain




class NowackiBeam(Function):
	r""" The Nowacki Beam Problem

	

	References
	----------
	.. [FSK08] Engineering Design via Surrogate Modeling: A Practical Guide
		Aleander I. J. Forrester, Andras Sobester, and Andy J. Keane
		Wiley, 2008
	"""

	def __init__(self):
		domain = build_beam_domain()
		funs = [beam_area, beam_bending]
		grads = [beam_area_grad, beam_bending_grad]
		Function.__init__(self, funs, domain, grads = grads, vectorized = True)		

def build_beam_domain():
	# Note the book has an invalid range for height "(20mm > b > 250 mm)" and breadth "(10 mm > b > 50mm)"
	# Here we follow the dimensions implied in the corresponding matlab code
	# b = x(1)*0.045 +0.005
	# h = x(2)*0.23 + 0.02
	# I assume these definitions are in meters 
	domain = BoxDomain([5e-3, 0.02], [50e-3,0.25], names = ['breadth (m)', 'height (m)'])
	return domain

def beam_area(X):
	X_shape = X.shape
	X = np.atleast_2d(X)
	area = X[:,0]*X[:,1]
	if len(X_shape) == 1:
		area = area[0]
	return area

def beam_area_grad(X):
	X_shape = X.shape
	X = np.atleast_2d(X)
	grad = np.array([X[:,1], X[:,0]]).T
	if len(X_shape) == 1:
		grad = grad[0]
	return grad


def beam_bending(X, F = 5e3, L = 1.5):
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

def beam_bending_grad(X, F = 5e3, L = 1.5):
	X_shape = X.shape
	X = np.atleast_2d(X)
	b = X[:,0]
	h = X[:,1]
	# From bending.m
	grad = np.array([-6*F*L/(b**2 * h**2), -2*6*F*L/(b * h**3)]).T
	if len(X_shape) == 1:
		grad = grad[0]
	return grad
		 
