# An example from https://www.sfu.ca/~ssurjano/otlcircuit.html

import numpy as np

# Hack to import domain
# https://stackoverflow.com/questions/6323860/sibling-package-imports
import sys, os
sys.path.insert(0, os.path.abspath('../../'))
from psdr import BoxDomain

__all__ = ['build_otl_circuit_domain', 'otl_circuit']

def build_otl_circuit_domain():
	# Parameters
	# R_b1, R_b2, R_f, R_c1, R_c2, beta
	lb = np.array([50, 25, 0.5, 1.2, 0.25, 50])
	ub = np.array([150, 70, 3, 2.5, 1.2, 300])

	return BoxDomain(lb, ub)

def otl_circuit(x, return_grad = False):
	if x.shape == 0:
		x = x.reshape(-1,6)
	Rb1 = x[:,0]
	Rb2 = x[:,1]
	Rf = x[:,2]
	Rc1 = x[:,3]
	Rc2 = x[:,4]
	beta = x[:,5]

	Vb1 = 12*Rb2/(Rb1 + Rb2)
	Vm = (Vb1 + 0.74)*beta*(Rc2 + 9)/(beta*(Rc2 + 9)+Rf) + 11.35*Rf/(beta*(Rc2 + 9) + Rf) + 0.74*Rf*beta*(Rc2 + 9)/( (beta*(Rc2 + 9) + Rf)*Rc1)
	if not return_grad: return Vm
	grad = np.vstack([
			-12*Rb2*beta*(Rc2 + 9)/((Rb1 + Rb2)**2*(Rf + beta*(Rc2 + 9))),
			12*Rb1*beta*(Rc2 + 9)/((Rb1 + Rb2)**2*(Rf + beta*(Rc2 + 9))),
			(-11.35*Rc1*Rf*(Rb1 + Rb2) - Rc1*beta*(0.74*Rb1 + 12.74*Rb2)*(Rc2 + 9) + 11.35*Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9)) - 0.74*Rf*beta*(Rb1 + Rb2)*(Rc2 + 9) + 0.74*beta*(Rb1 + Rb2)*(Rc2 + 9)*(Rf + beta*(Rc2 + 9)))/(Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9))**2),
			-0.74*Rf*beta*(Rc2 + 9)/(Rc1**2*(Rf + beta*(Rc2 + 9))),
			beta*(-11.35*Rc1*Rf*(Rb1 + Rb2) - Rc1*beta*(0.74*Rb1 + 12.74*Rb2)*(Rc2 + 9) + Rc1*(0.74*Rb1 + 12.74*Rb2)*(Rf + beta*(Rc2 + 9)) - 0.74*Rf*beta*(Rb1 + Rb2)*(Rc2 + 9) + 0.74*Rf*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9)))/(Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9))**2),
			(Rc2 + 9)*(-11.35*Rc1*Rf*(Rb1 + Rb2) - Rc1*beta*(0.74*Rb1 + 12.74*Rb2)*(Rc2 + 9) + Rc1*(0.74*Rb1 + 12.74*Rb2)*(Rf + beta*(Rc2 + 9)) - 0.74*Rf*beta*(Rb1 + Rb2)*(Rc2 + 9) + 0.74*Rf*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9)))/(Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9))**2),
		]).T
	return Vm, grad
