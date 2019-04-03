# An example from https://www.sfu.ca/~ssurjano/otlcircuit.html

import numpy as np
from psdr import BoxDomain, Function

__all__ = ['build_otl_circuit_domain', 'otl_circuit', 'otl_circuit_grad', 'OTLCircuit']

class OTLCircuit(Function):
	r"""
	"""
	def __init__(self):
		domain = build_otl_circuit_domain()
		funs = [otl_circuit]
		grads = [otl_circuit_grad] 

		Function.__init__(self, funs, domain, grads = grads, vectorized = True)


def build_otl_circuit_domain():
	# Parameters
	# R_b1, R_b2, R_f, R_c1, R_c2, beta
	lb = np.array([50, 25, 0.5, 1.2, 0.25, 50])
	ub = np.array([150, 70, 3, 2.5, 1.2, 300])

	return BoxDomain(lb, ub, names = ['R_b1', 'R_b2', 'R_f', 'R_c1', 'R_c2', 'beta'])

def otl_circuit(x, return_grad = False):
	import numpy as np
	x = np.array(x).reshape(-1,6)
	
	Rb1 = x[:,0]
	Rb2 = x[:,1]
	Rf = x[:,2]
	Rc1 = x[:,3]
	Rc2 = x[:,4]
	beta = x[:,5]

	Vb1 = 12*Rb2/(Rb1 + Rb2)
	Vm = (Vb1 + 0.74)*beta*(Rc2 + 9)/(beta*(Rc2 + 9)+Rf) + 11.35*Rf/(beta*(Rc2 + 9) + Rf) + 0.74*Rf*beta*(Rc2 + 9)/( (beta*(Rc2 + 9) + Rf)*Rc1)
	
	return Vm

def otl_circuit_grad(x):
	x = x.reshape(-1,6)
	Rb1 = x[:,0]
	Rb2 = x[:,1]
	Rf = x[:,2]
	Rc1 = x[:,3]
	Rc2 = x[:,4]
	beta = x[:,5]

	grad = np.vstack([
			-12*Rb2*beta*(Rc2 + 9)/((Rb1 + Rb2)**2*(Rf + beta*(Rc2 + 9))),
			12*Rb1*beta*(Rc2 + 9)/((Rb1 + Rb2)**2*(Rf + beta*(Rc2 + 9))),
			(-11.35*Rc1*Rf*(Rb1 + Rb2) - Rc1*beta*(0.74*Rb1 + 12.74*Rb2)*(Rc2 + 9) + 11.35*Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9)) - 0.74*Rf*beta*(Rb1 + Rb2)*(Rc2 + 9) + 0.74*beta*(Rb1 + Rb2)*(Rc2 + 9)*(Rf + beta*(Rc2 + 9)))/(Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9))**2),
			-0.74*Rf*beta*(Rc2 + 9)/(Rc1**2*(Rf + beta*(Rc2 + 9))),
			beta*(-11.35*Rc1*Rf*(Rb1 + Rb2) - Rc1*beta*(0.74*Rb1 + 12.74*Rb2)*(Rc2 + 9) + Rc1*(0.74*Rb1 + 12.74*Rb2)*(Rf + beta*(Rc2 + 9)) - 0.74*Rf*beta*(Rb1 + Rb2)*(Rc2 + 9) + 0.74*Rf*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9)))/(Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9))**2),
			(Rc2 + 9)*(-11.35*Rc1*Rf*(Rb1 + Rb2) - Rc1*beta*(0.74*Rb1 + 12.74*Rb2)*(Rc2 + 9) + Rc1*(0.74*Rb1 + 12.74*Rb2)*(Rf + beta*(Rc2 + 9)) - 0.74*Rf*beta*(Rb1 + Rb2)*(Rc2 + 9) + 0.74*Rf*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9)))/(Rc1*(Rb1 + Rb2)*(Rf + beta*(Rc2 + 9))**2),
		]).T
	return grad



