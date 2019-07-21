from __future__ import division

import numpy as np
from psdr import Function, BoxDomain


class HartmannMHD(Function):
	r""" The Hartmann test problem from the MHD community

	This problem is largely taken from [GCSW17]_ with one modification;
	The second parameter 'fluid density' does not appear in either function
	so we do not include this parameter in the domain. 


	References
	---------- 
	.. [GCSW17] "Dimension reduction in magnetohydrodynamics power generation models: Dimensional analysis and active subspaces"
		Andrew Glaws, Paul G. Constantine, John N. Shadid, and Timothy M. Wildey
		Stat Anal Data Min. 2017, Vol 10, pp 312--325, DOI:10.1002/sam.11355
	"""

	def __init__(self):
		domain = build_hartmann_domain()
		funs = [u_ave, Bind]
		grads = [u_ave_grad, Bind_grad]
		Function.__init__(self, funs, domain, grads = grads, vectorized = True)

	def __str__(self):
		return "<Hartmann MHD Test Functions>"


def build_hartmann_domain():
	# Ranges are taken from GCSW17, Table 2
	# The second parameter 'fluid density' does not appear in either function
	return BoxDomain([0.05, 0.5, 0.5, 0.1], [0.2, 3, 3, 1], 
		names = [
			'fluid viscosity',				# mu
			'applied pressure gradient',	# d p_0/ dx
			'resistivity',					# eta
			'applied magnetic field'		# B_0
		])


def u_ave(X):
	X = np.atleast_2d(X)
	mu = X[:,0]
	dpdx = X[:,1]
	eta = X[:,2]
	B0 = X[:,3]
	ell = 1 # defined right under subsec 4.1 in GCSW17

	u_ave = -dpdx*eta/B0**2*(1 - B0*ell/np.sqrt(eta*mu)/np.tanh(B0*ell/np.sqrt(eta*mu)))

	return u_ave

def u_ave_grad(X):
	X = np.atleast_2d(X)
	mu = X[:,0]
	dpdx = X[:,1]
	eta = X[:,2]
	B0 = X[:,3]
	ell = 1 # defined right under subsec 4.1 in GCSW17

	return np.vstack([
		-dpdx*eta*(-B0**2*ell**2*1/(eta*mu)/(2*mu*np.sinh(B0*ell/np.sqrt(eta*mu))**2) + B0*ell/np.tanh(B0*ell/np.sqrt(eta*mu))/(2*mu*np.sqrt(eta*mu)))/B0**2,
		-eta*(-B0*ell/np.tanh(B0*ell/np.sqrt(eta*mu))/np.sqrt(eta*mu) + 1)/B0**2,
		-dpdx*eta*(-B0**2*ell**2*1/(eta*mu)/(2*eta*np.sinh(B0*ell/np.sqrt(eta*mu))**2) + B0*ell/np.tanh(B0*ell/np.sqrt(eta*mu))/(2*eta*np.sqrt(eta*mu)))/B0**2 - dpdx*(-B0*ell/np.tanh(B0*ell/np.sqrt(eta*mu))/np.sqrt(eta*mu) + 1)/B0**2,
		-dpdx*eta*(B0*ell**2*(1/(eta*mu))/np.sinh(B0*ell/np.sqrt(eta*mu))**2 - ell/np.tanh(B0*ell/np.sqrt(eta*mu))/np.sqrt(eta*mu))/B0**2 + 2*dpdx*eta*(-B0*ell/np.tanh(B0*ell/np.sqrt(eta*mu))/np.sqrt(eta*mu) + 1)/B0**3
	]).T	


def Bind(X):
	X = np.atleast_2d(X)
	mu = X[:,0]
	dpdx = X[:,1]
	eta = X[:,2]
	B0 = X[:,3]
	ell = 1 # defined right under subsec 4.1 in GCSW17
	mu0 = 1 
	Bind = dpdx*(ell*mu0/(2*B0))*(1 - 2*np.sqrt(eta*mu)/(B0*ell)*np.tanh(B0*ell/(2*np.sqrt(eta*mu))))
	return Bind


def Bind_grad(X):
	X = np.atleast_2d(X)
	mu = X[:,0]
	dpdx = X[:,1]
	eta = X[:,2]
	B0 = X[:,3]
	ell = 1 # defined right under subsec 4.1 in GCSW17
	mu0 = 1

	return np.vstack([
		dpdx*ell*mu0*((-np.tanh(B0*ell/(2*np.sqrt(eta*mu)))**2 + 1)/(2*mu) - np.sqrt(eta*mu)*np.tanh(B0*ell/(2*np.sqrt(eta*mu)))/(B0*ell*mu))/(2*B0),
		ell*mu0*(1 - 2*np.sqrt(eta*mu)*np.tanh(B0*ell/(2*np.sqrt(eta*mu)))/(B0*ell))/(2*B0),
		dpdx*ell*mu0*((-np.tanh(B0*ell/(2*np.sqrt(eta*mu)))**2 + 1)/(2*eta) - np.sqrt(eta*mu)*np.tanh(B0*ell/(2*np.sqrt(eta*mu)))/(B0*ell*eta))/(2*B0),
		dpdx*ell*mu0*(-(-np.tanh(B0*ell/(2*np.sqrt(eta*mu)))**2 + 1)/B0 + 2*np.sqrt(eta*mu)*np.tanh(B0*ell/(2*np.sqrt(eta*mu)))/(B0**2*ell))/(2*B0) - dpdx*ell*mu0*(1 - 2*np.sqrt(eta*mu)*np.tanh(B0*ell/(2*np.sqrt(eta*mu)))/(B0*ell))/(2*B0**2)
	]).T 
