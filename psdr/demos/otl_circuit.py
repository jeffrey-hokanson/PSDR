# An example from https://www.sfu.ca/~ssurjano/otlcircuit.html
import numpy as np
from .. import BoxDomain

__all__ = ['build_otl_circuit_domain', 'otl_circuit']

def build_otl_circuit_domain():
	# Parameters
	# R_b1, R_b2, R_f, R_c1, R_c2, beta
	lb = np.array([50, 25, 0.5, 1.2, 0.25, 50])
	ub = np.array([150, 70, 3, 2.5, 1.2, 300])

	return BoxDomain(lb, ub)

def otl_circuit(x):
	x = x.reshape(-1,6)
	Rb1 = x[:,0]
	Rb2 = x[:,1]
	Rf = x[:,2]
	Rc1 = x[:,3]
	Rc2 = x[:,4]
	beta = x[:,5]

	Vb1 = 12*Rb2/(Rb1 + Rb2)
	Vm = (Vb1 + 0.74)*beta*(Rc2 + 9)/(beta*(Rc2 + 9)+Rf) + 11.35*Rf/(beta*(Rc2 + 9) + Rf) + 0.74*Rf*beta*(Rc2 + 9)/( (beta*(Rc2 + 9) + Rf)*Rc1)
	return Vm
