# An example from https://www.sfu.ca/~ssurjano/otlcircuit.html

import numpy as np
from psdr import BoxDomain, Function

__all__ = ['build_otl_circuit_domain', 'otl_circuit', 'otl_circuit_grad', 'OTLCircuit']

class OTLCircuit(Function):
	r""" The OTL circuit test function

	The OTL Circuit function "models an output transformerless push pull circuit" [VLSE]_

	.. math::
	
		f(R_{b1}, R_{b2}, R_f, R_{c1}, R_{c2}, \beta) :=&
			\frac{ \beta (V_{b1} + 0.74)(R_{c2} + 9)}{\beta(R_{c2} + 9) + R_f}
			+ \frac{11.35 R_f}{\beta(R_{c2} + 9) + R_f} \\
			&+\frac{0.74 R_f \beta (R_{c2} + 9)}{R_{c1}(\beta(R_{c2} + 9) + R_f)}, \\
		\text{where } V_{b1} :=& \frac{12 R_{b2}}{R_{b1} + R_{b2}}
	
	====================================    ========================
	Variable                                Interpretation
	====================================    ========================
	:math:`R_{b1} \in [50, 150]`			resistance b1 (K-Ohms)
	:math:`R_{b2} \in [25, 75]`				resistance b2 (K-Ohms)
	:math:`R_{f} \in [0.5, 3]`				resistance f (K-Ohms)
	:math:`R_{c1} \in [1.2, 2.5]`			resistance c1 (K-Ohms)
	:math:`R_{c1} \in [0.25, 1.2]`			resistance c2 (K-Ohms)
	:math:`\beta \in [50, 300]`				current gain (Amperes)
	====================================    ========================

	Parameters
	----------
	dask_client: dask.distributed.Client or None
		If specified, allows distributed computation with this function.



	References
	----------
	.. [VLSE] Virtual Library of Simulation Experiments, OTL Circuit
		https://www.sfu.ca/~ssurjano/otlcircuit.html

	"""
	def __init__(self, dask_client = None):
		domain = build_otl_circuit_domain()
		funs = [otl_circuit]
		grads = [otl_circuit_grad] 

		Function.__init__(self, funs, domain, grads = grads, vectorized = True, dask_client = dask_client)

	def __str__(self):
		return "<OTL Circuit Function>"

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



