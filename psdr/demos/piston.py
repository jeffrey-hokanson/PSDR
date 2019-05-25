# Piston function from https://www.sfu.ca/~ssurjano/piston.html
import numpy as np

from psdr import BoxDomain, Function

__all__ = ['build_piston_domain', 'piston', 'Piston']


class Piston(Function):
	r""" Piston test function

	The Piston function "models the circular motion of a piston within a cylinder" [VLSE_piston]_.
	
	.. math::

		f(M, S, V_0, k, P_0, T_a, T_0) :=&
			2\pi \sqrt{\frac{M}{k+ S^2 \frac{P_0V_0}{T_0} \frac{T_a}{V^2}}}, \text{ where } \\
		V &= \frac{S}{2k}\left(  
				\sqrt{A^2 + 4k \frac{P_0V_0}{T_0}T_a} - A
			\right) \\
		A &= P_0 S + 19.62M - \frac{k V_0}{S}
	

	================================================ 	========================
	Variable                               			 	Interpretation
	================================================ 	========================
	:math:`M \in [30, 60]`		            			piston weight (kg)
	:math:`S \in [0.005, 0.020]`	        			piston surface area (m^2)
	:math:`V_0 \in [0.002, 0.010]`	        			initial gas volume (m^3)
	:math:`k \in [1000, 5000]`							spring coefficient (N/m)
	:math:`P_0 \in [90\times 10^3, 110 \times 10^3]` 	atmospheric pressure (N/m^2)
	:math:`T_a \in [290, 296]`							ambient temperature (K)
	:math:`T_0 \in [340, 360]`							filling gas temperature (K)
	================================================	========================
	

	Parameters
	----------
	dask_client: dask.distributed.Client or None
		If specified, allows distributed computation with this function.
	
	References
	----------
	.. [VLSE_piston] Virtual Library of Simulation Experiments, Piston Function
		https://www.sfu.ca/~ssurjano/piston.html

	"""
	def __init__(self, dask_client = None):
		domain = build_piston_domain()
		funs = [piston]
		grads = [piston_grad]

		Function.__init__(self, funs, domain, grads = grads, vectorized = True, dask_client = dask_client)

	def __str__(self):
		return "<Piston Function>"

def build_piston_domain():
	# Parameters
	# M S V0 k P0 Ta T0
	lb = np.array([30, 0.005, 0.002, 1000, 90e3, 290, 340])
	ub = np.array([60, 0.020, 0.010, 5000, 110e3, 296, 360])
	return BoxDomain(lb, ub, names = ['M', 'S', 'V_0', 'k', 'P_0', 'T_a', 'T_0'])


def piston(X):
	""" Piston test function

	"""
	import numpy as np
	X = X.reshape(-1, 7)

	# Split the variables
	M  = X[:,0]
	S  = X[:,1]
	V0 = X[:,2]
	k  = X[:,3]
	P0 = X[:,4]
	Ta = X[:,5]
	T0 = X[:,6]

	A = P0*S + 19.62*M - k*V0/S
	V = S/(2*k)*( np.sqrt(A**2 + 4*k*P0*V0/T0*Ta) - A)
	C = 2*np.pi*np.sqrt(M/(k+S**2*(P0*V0/T0)*(Ta/V**2)))

	return C

def piston_grad(X):
	import numpy as np
	X = X.reshape(-1,7)
	# Split the variables
	M  = X[:,0]
	S  = X[:,1]
	V0 = X[:,2]
	k  = X[:,3]
	P0 = X[:,4]
	Ta = X[:,5]
	T0 = X[:,6]
	
	# Compute analytic gradients determined using Sympy
	grad = np.vstack([
			2*np.pi*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k)*(-2*M*P0*Ta*V0*k**2*(39.24 - 2*(384.9444*M + 19.62*P0*S - 19.62*V0*k/S)/np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2))/(T0*(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k)**2*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3) + 1/(2*(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k)))/M,
			-4*np.pi*P0*Ta*V0*k**2*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(2*P0 - (2*P0 + 2*V0*k/S**2)*(19.62*M + P0*S - V0*k/S)/np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + 2*V0*k/S**2)/(T0*(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k)*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3),
			np.pi*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(-4*P0*Ta*V0*k**2*(-2*(2*P0*Ta*k/T0 - k*(19.62*M + P0*S - V0*k/S)/S)/np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) - 2*k/S)/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3) - 4*P0*Ta*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2))/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k),
			np.pi*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(-4*P0*Ta*V0*k**2*(-2*(2*P0*Ta*V0/T0 - V0*(19.62*M + P0*S - V0*k/S)/S)/np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) - 2*V0/S)/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3) - 8*P0*Ta*V0*k/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) - 1)/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k),
			np.pi*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(-4*P0*Ta*V0*k**2*(2*S - 2*(S*(19.62*M + P0*S - V0*k/S) + 2*Ta*V0*k/T0)/np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2))/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3) - 4*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2))/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k),
			np.pi*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(16*P0**2*Ta*V0**2*k**3/(T0**2*np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2)*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3) - 4*P0*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2))/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k),
			np.pi*np.sqrt(M/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k))*(-16*P0**2*Ta**2*V0**2*k**3/(T0**3*np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2)*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**3) + 4*P0*Ta*V0*k**2/(T0**2*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2))/(4*P0*Ta*V0*k**2/(T0*(-19.62*M - P0*S + np.sqrt(4*P0*Ta*V0*k/T0 + (19.62*M + P0*S - V0*k/S)**2) + V0*k/S)**2) + k),
		]).T
	return grad 
