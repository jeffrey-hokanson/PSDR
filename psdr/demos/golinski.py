#! /usr/bin/env python
import numpy as np

from psdr import BoxDomain, NormalDomain, TensorProductDomain, Function

__all__ = ['golinski_volume', 
	'golinski_constraint1',
	'golinski_constraint2',
	'golinski_constraint3',
	'golinski_constraint4',
	'golinski_constraint5',
	'golinski_constraint6',
	'golinski_constraint7',
	'golinski_constraint8',
	'golinski_constraint9',
	'golinski_constraint24',
	'golinski_constraint25',
	'build_golinski_design_domain', 
	'build_golinski_random_domain',
	'GolinskiGearbox',
	]


class GolinskiGearbox(Function):
	r""" The Golinski Gearbox Optimization Test Problem


	This test problem originally descibed by Golinski [Gol70]_ and subsequently 
	appearing in, for example, [Ray03]_ and [MDO]_,
	seeks to design a gearbox (speedreducer) to minimize volume subject to a number of constraints.

	Here we take our formulation following [Ray03]_, which reduces the original 25 constraints to 11
	by removing redundancy. We also shift these constraints such that they are satisfied if the values
	returned are negative and the constraints are violated if their value is positive.

	Parameters
	----------
	dask_client: dask.distributed.Client or None
		If specified, allows distributed computation with this function.


	References
	----------
	.. [MDO] Langley Research Center: Multidisciplinary Optimization Test Suite,
		`Golinski's Speed Reducer <http://www.eng.buffalo.edu/Research/MODEL/mdo.test.orig/class2prob4.html>`_	

	.. [Gol70] "Optimal Synthesis Problems Solved by Means of Nonlinear Programming and Random Methods",
		Jan Golinski, J. Mech. 5, 1970, pp.287--309.
		https://doi.org/10.1016/0022-2569(70)90064-9

	.. [Ray03] "Golinski's Speed Reducer Problem Revisited", Tapabrata Ray, AIAA Journal,
			41(3), 2003, pp 556--558
			https://doi.org/10.2514/2.1984

	"""
	def __init__(self, dask_client = None):
		domain = build_golinski_design_domain() 
		funs = [golinski_volume,
				golinski_constraint1,
				golinski_constraint2,
				golinski_constraint3,
				golinski_constraint4,
				golinski_constraint5,
				golinski_constraint6,
				golinski_constraint7,
				golinski_constraint8,
				golinski_constraint9,
				golinski_constraint24,
				golinski_constraint25,
				]	
		grads = [golinski_volume_grad,
				golinski_constraint1_grad,
				golinski_constraint2_grad,
				golinski_constraint3_grad,
				golinski_constraint4_grad,
				golinski_constraint5_grad,
				golinski_constraint6_grad,
				golinski_constraint7_grad,
				golinski_constraint8_grad,
				golinski_constraint9_grad,
				golinski_constraint24_grad,
				golinski_constraint25_grad,
				]
		Function.__init__(self, funs, domain, grads = grads, vectorized = True, dask_client = dask_client)
	
	def __str__(self):
		return "<Golinski Gearbox Function>"


def expand_variables(x):
	try:
		if len(x.shape) == 1:
			x = x.reshape(1,-1)
	except AttributeError:
		pass

	x = x.T

	x1 = x[0]
	x2 = x[1]
	x3 = np.ones(x[0].shape)*17 # This is the troublesome integer value
	x4 = x[2]
	x5 = x[3]
	x6 = x[4]
	x7 = x[5]
	#if len(x) > 6:
	#	# We do not use += because if these inputs are vectors, 
	#	# x1 will be the first row of x, and += will modify that original row
	#	x1 = x1 + x[6]
	#	x2 = x2 + x[7]
	#	x4 = x4 + x[8]
	#	x5 = x5 + x[9]
	#	x6 = x6 + x[10]
	#	x7 = x7 + x[11]
	return x1, x2, x3, x4, x5, x6, x7



def golinski_volume(x, return_grad = False):
	""" Volume (objective function) for Golinski Gearbox test problem
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)

	return 0.7854*x1*x2**2*( 3.3333*x3**2 + 14.9334*x3 - 43.0934) \
		-1.5079*x1*(x6**2 + x7**2) + 7.477*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)
	

def golinski_volume_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
		
	# Gradient as computed symbolically using Sympy
	return np.vstack([
			0.7854*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.5079*x6**2 - 1.5079*x7**2,
			1.5708*x1*x2*(3.3333*x3**2 + 14.9334*x3 - 43.0934),
			0.7854*x6**2,
			0.7854*x7**2,
			-3.0158*x1*x6 + 1.5708*x4*x6 + 22.431*x6**2,
			-3.0158*x1*x7 + 1.5708*x5*x7 + 22.431*x7**2,]).T
	

def golinski_constraint1(x):
	""" First constraint from the Golinski Gearbox problem
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 27/(x1*x2**2*x3) - 1

def golinski_constraint1_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			-27/(x1**2*x2**2*x3),
			-54/(x1*x2**3*x3),
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T

def golinski_constraint2(x):
	"""Second constraint from the Golinski Gearbox problem
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 397.5/(x1*x2**2*x3**2) - 1

def golinski_constraint2_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	
	return np.vstack([
			-397.5/(x1**2*x2**2*x3**2),
			-795.0/(x1*x2**3*x3**2),
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T

def golinski_constraint3(x):
	"""Third constraint from the Golinski Gearbox problem
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 1.93/(x2*x3*x6**4)*x4**3 - 1.

def golinski_constraint3_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			0*x1,
			-1.93*x4**3/(x2**2*x3*x6**4),
			5.79*x4**2/(x2*x3*x6**4),
			0*x1,
			-7.72*x4**3/(x2*x3*x6**5),
			0*x1,
		]).T
	

def golinski_constraint4(x):
	"""Fourth constraint from the Golinski Gearbox problem
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 1.93/(x2*x3*x7**4)*x5**3 - 1.

def golinski_constraint4_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			0*x1,
			-1.93*x5**3/(x2**2*x3*x7**4),
			0*x1,
			5.79*x5**2/(x2*x3*x7**4),
			0*x1,
			-7.72*x5**3/(x2*x3*x7**5),	
		]).T
	

def golinski_constraint5(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.sqrt( (745*x4/x2/x3)**2 + 16.9e6)/(110.0*x6**3) - 1.
	
def golinski_constraint5_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			0*x1,
			-5045.68181818182*x4**2/(x2**3*x3**2*x6**3*np.sqrt(16900000.0 + 555025*x4**2/(x2**2*x3**2))),
			5045.68181818182*x4/(x2**2*x3**2*x6**3*np.sqrt(16900000.0 + 555025*x4**2/(x2**2*x3**2))),
			0*x1,
			-0.0272727272727273*np.sqrt(16900000.0 + 555025*x4**2/(x2**2*x3**2))/x6**4,
			0*x1,
		]).T

def golinski_constraint6(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.sqrt( (745*x5/x2/x3)**2 + 157.5e6)/(85.0*x7**3) - 1.

def golinski_constraint6_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			0*x1, 
			-6529.70588235294*x5**2/(x2**3*x3**2*x7**3*np.sqrt(157500000.0 + 555025*x5**2/(x2**2*x3**2))),
			0*x1, 
			6529.70588235294*x5/(x2**2*x3**2*x7**3*np.sqrt(157500000.0 + 555025*x5**2/(x2**2*x3**2))),
			0*x1,
			-0.0352941176470588*np.sqrt(157500000.0 + 555025*x5**2/(x2**2*x3**2))/x7**4,
		]).T

def golinski_constraint7(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return x2*x3/40 - 1.

def golinski_constraint7_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return  np.vstack([
			0*x1,
			x3/40,
			0*x1,	# This is constant because it is the integer value
			0*x1,
			0*x1,
			0*x1,
		]).T

def golinski_constraint8(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 5*x2/x1 - 1.

def golinski_constraint8_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return  np.vstack([
			-5*x2/x1**2,
			5/x1,
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T

def golinski_constraint9(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return x1/(12*x2) - 1.

def golinski_constraint9_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			1/(12*x2),
			-x1/(12*x2**2),
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T

def golinski_constraint24(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return (1.5*x6 + 1.9)/x4 - 1.

def golinski_constraint24_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			0*x1,
			0*x1,
			-(1.5*x6 + 1.9)/x4**2,
			0*x1,
			1.5/x4,
			0*x1,
		]).T
	

def golinski_constraint25(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return (1.1*x7 + 1.9)/x5 - 1.
	
def golinski_constraint25_grad(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.vstack([
			0*x1,
			0*x1,
			0*x1,
			-(1.1*x7 + 1.9)/x5**2,
			0*x1,
			1.1/x5,
		]).T
	return val, grad

# Units of cm
def build_golinski_design_domain():
	return BoxDomain([2.6,0.7, 7.3, 7.3, 2.9, 5.0], [3.6, 0.8, 8.3, 8.3, 3.9, 5.5], 
		names = ["width of gear face", 
				"teeth module", 
				"shaft 1 length between bearings",
				"shaft 2 length between bearings",
				"diameter of shaft 1",
				"diameter of shaft 2"]
		 )


# Taken from table 3 of Hu, Zhou, Chen, Parks, 2017 in AIAA journal 
def build_golinski_random_domain(clip = None):
	return TensorProductDomain([NormalDomain(0,21e-4**2, clip = clip),
				NormalDomain(0, 1e-4**2, clip = clip),
				NormalDomain(0, 30e-4**2, clip = clip),
				NormalDomain(0, 30e-4**2, clip = clip),
				NormalDomain(0, 21e-4**2, clip = clip),
				NormalDomain(0, 30e-4**2, clip = clip)])


if __name__ == '__main__':
	# If called, act as a Dakota interface to this function
	import dakota.interfacing as di

	gg = GolinskiGearbox()
	
	params, results = di.read_parameters_file()
	x = np.array([params['x%d' % (i+1) ] for i in range(6)])
	y = gg(x)
	results['f'].function = y[0]
	for i in range(1,12):
		results['c%d' % i ].function = y[i]	

	results.write()



