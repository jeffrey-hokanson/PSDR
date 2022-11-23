import numpy as np

from psdr import BoxDomain, NormalDomain, Function, LogNormalDomain, UniformDomain, TensorProductDomain


__all__ = ['build_borehole_domain', 'build_borehole_uncertain_domain', 'borehole', 'borehole_grad', 'Borehole']

class Borehole(Function):
	r""" The borehole test function

	A function implementing the borehole test function [VLSE_borehole]_.
	This function has the form

	.. math::
	
		f(r_w, r, T_u, H_u, T_l, H_l, L, K_w) =
			\frac{ 2\pi T_u (H_u - H_l)}{ 
				\ln(r/r_w) 
				\left(   
					1 + \frac{2 L T_u}{\ln(r/r_w) r_w^2 K_w} + \frac{T_u}{T_l}
				\right)
			}

	where the input variables have the domain

	====================================    ========================
	Variable                                Interpretation
	====================================    ========================
	:math:`r_w\in [0.05, 0.15]`             radius of borehole (m)    
	:math:`r \in [100, 50 \times 10^3]`     radius of influence (m)
	:math:`T_u \in [63070,115600]`          trasmissivity of upper aquifer (m^2/yr)
	:math:`H_u \in [ 990, 1110]`			potentiometric head of upper aquifer (m)
	:math:`T_l \in [63.1, 116]` 			transmissivity of lower aquifer (m^2/yr)
	:math:`H_l \in [700, 820]` 				potentiometric head of lower aquifer (m)
	:math:`L \in [1120, 1680]`				length of borehole (m)
	:math:`K_w \in  [9855, 12045]`			hydraulic conductivity of borehole (m/yr)
	====================================    ========================

	An alternative to this deterministic domain is an uncertain domain where
	:math:`r_w \sim \mathcal{N}(0.10, 0.0161812)` and :math:`\log r \sim \mathcal{N}(7.71, 1.0056)`
	and the remainder come from a uniform distribution on the domain previously specified.


	Parameters
	----------
	domain: ['deterministic', 'uncertain']
		Which domain to use when constructing the function
	dask_client: dask.distributed.Client or None
		If specified, allows distributed computation with this function.


	References
	----------
	.. [VLSE_borehole] Virtual Library of Simulation Experiments, Borehole Function
		 https://www.sfu.ca/~ssurjano/borehole.html 

	"""

	def __init__(self, domain = 'deterministic', dask_client = None):
		assert domain in ['deterministic', 'uncertain']
		if domain == 'deterministic':
			domain = build_borehole_domain()
		else:
			domain = build_borehole_uncertain_domain()

		funs = [borehole]
		grads = [borehole_grad]

		Function.__init__(self, funs, domain, grads = grads, vectorized = True, dask_client = dask_client)

	def __str__(self):
		return "<Borehole Function>"

def build_borehole_domain():
	r""" Constructs a deterministic domain associated with the borehole function

	Returns
	-------
	dom: BoxDomain
		Domain associated with the borehole function
	"""
	# Parameters
	# r_w, r, T_u, H_u, T_l, H_l, L, K_w
	lb = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
	ub = np.array([0.15, 50e3, 115600, 1110, 116, 820, 1680, 12045])

	return BoxDomain(lb, ub, names = ['r_w', 'r', 'T_u', 'H_u', 'T_l', 'H_l', 'L', 'K_w'])

def build_borehole_uncertain_domain():
	r""" Constructs an uncertain domain associated with the borehole function

	Returns
	-------
	dom: TensorProductDomain
		Uncertain domain associated with the borehole function
	"""
	return TensorProductDomain([
		NormalDomain(0.10, 0.0161812**2, names = 'r_w'),
		LogNormalDomain(7.71, 1.0056**2, names = 'r'),
		UniformDomain(63070, 115600, names = 'T_u'), 
		UniformDomain(990, 1110, names = 'H_u'),
		UniformDomain(63.1, 116, names = 'T_l'),
		UniformDomain(700, 820, names = 'H_l'),
		UniformDomain(1120, 1680, names = 'L'),
		UniformDomain(9855, 12045, names = 'K_w')
	])


def borehole(X):
	""" The borehole test function

	See description in :meth:`psdr.demos.Borehole`
	
	Parameters
	----------
	X: array-like (?, 8)
		Input in application units
	
	Return
	------
	y: np.ndarray (?,)
		Output of borehole function
	"""
	import numpy as np
	X = X.reshape(-1, 8)

	# Split the variables
	r_w = X[:,0]
	r   = X[:,1]
	T_u = X[:,2]
	H_u = X[:,3]
	T_l = X[:,4]
	H_l = X[:,5]
	L   = X[:,6]
	K_w = X[:,7]

	val = 2*np.pi*T_u*(H_u - H_l)/(np.log(r/r_w)*(1 + 2*L*T_u/(np.log(r/r_w)*r_w**2*K_w) + T_u/T_l))
	return val

def borehole_grad(X):
	""" The borehole test function gradient

	See description in :meth:`psdr.demos.Borehole`
	
	Parameters
	----------
	X: array-like (?, 8)
		Input in application units
	
	Return
	------
	y: np.ndarray (?,8)
		Gradient of borehole test function
	"""
	import numpy as np
	X = np.atleast_2d(np.array(X))

	# Split the variables
	r_w = X[:,0]
	r   = X[:,1]
	T_u = X[:,2]
	H_u = X[:,3]
	T_l = X[:,4]
	H_l = X[:,5]
	L   = X[:,6]
	K_w = X[:,7]

	# Gradient computed analytically using Sympy
	grad = np.vstack([
		-2*np.pi*K_w*T_l*T_u*r_w*(H_l - H_u)*(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u*(2*np.log(r/r_w) - 1) + 2*L*T_l*T_u)/((K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u)**2*np.log(r/r_w)),
		2*np.pi*K_w**2*T_l*T_u*r_w**4*(H_l - H_u)*(T_l + T_u)/(r*(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u)**2),
		2*np.pi*K_w*T_l*r_w**2*(H_l - H_u)*(-K_w*T_l*r_w**2*np.log(r/r_w) - K_w*T_u*r_w**2*np.log(r/r_w) - 2*L*T_l*T_u + T_u*(K_w*r_w**2*np.log(r/r_w) + 2*L*T_l))/(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u)**2,
		2*np.pi*K_w*T_l*T_u*r_w**2/(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u),
		-2*np.pi*K_w**2*T_u**2*r_w**4*(H_l - H_u)*np.log(r/r_w)/(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u)**2,
		-2*np.pi*K_w*T_l*T_u*r_w**2/(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u),
		4*np.pi*K_w*T_l**2*T_u**2*r_w**2*(H_l - H_u)/(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u)**2,
		-4*np.pi*L*T_l**2*T_u**2*r_w**2*(H_l - H_u)/(K_w*T_l*r_w**2*np.log(r/r_w) + K_w*T_u*r_w**2*np.log(r/r_w) + 2*L*T_l*T_u)**2,
		]).T
	return grad
