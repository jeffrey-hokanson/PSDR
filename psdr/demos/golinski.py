import numpy as np
from domains import BoxDomain, NormalDomain, ComboDomain

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
	]
	

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
	if len(x) > 6:
		# We do not use += because if these inputs are vectors, 
		# x1 will be the first row of x, and += will modify that original row
		x1 = x1 + x[6]
		x2 = x2 + x[7]
		x4 = x4 + x[8]
		x5 = x5 + x[9]
		x6 = x6 + x[10]
		x7 = x7 + x[11]
	return x1, x2, x3, x4, x5, x6, x7


def golinski_volume(x, return_grad = False):
	""" Volume (objective function) for Golinski Gearbox test problem

	Objective function from the Golinski Gearbox test problem [Gol70].
	Here we use the formulas provided by [Ray03]. 
	Originally posed over seven variables where 6 were real numbers and another, x3,
	is an integer.  In this code we fix x3 = 17 to avoid issues with integer domains.  

	[Gol70]: "Optimal Synthesis Problems Solved by Means of Nonlinear Programming and Random Methods",
		Jan Golinski, J. Mech. 5, 1970, pp.287--309

	[Ray03]: "Golinski's Speed Reducer Problem Revisited", Tapabrata Ray, AIAA Journal,
			41(3), 2003, pp 556--558

	Parameters
	----------
	x: np.ndarray (M, 6)
		Input parameters 

	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)

	val = 0.7854*x1*x2**2*( 3.3333*x3**2 + 14.9334*x3 - 43.0934) \
		-1.5079*x1*(x6**2 + x7**2) + 7.477*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)

	if not return_grad:
		return val
	else:
		# Gradient as computed symbolically using Sympy
		grad = np.vstack([
			0.7854*x2**2*(3.3333*x3**2 + 14.9334*x3 - 43.0934) - 1.5079*x6**2 - 1.5079*x7**2,
			1.5708*x1*x2*(3.3333*x3**2 + 14.9334*x3 - 43.0934),
			0.7854*x6**2,
			0.7854*x7**2,
			-3.0158*x1*x6 + 1.5708*x4*x6 + 22.431*x6**2,
			-3.0158*x1*x7 + 1.5708*x5*x7 + 22.431*x7**2,]).T
		return val, grad	

# These are taken from Ray 2003, AIAA Journal
def golinski_constraint1(x, return_grad = False):
	""" First constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.

	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - 27/(x1*x2**2*x3) 
	if not return_grad:
		return val
	grad = np.vstack([
			27/(x1**2*x2**2*x3),
			54/(x1*x2**3*x3),
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T
	return val, grad

def golinski_constraint2(x, return_grad = False):
	"""Second constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - 397.5/(x1*x2**2*x3**2)
	if not return_grad: return val
	grad = np.vstack([
			397.5/(x1**2*x2**2*x3**2),
			795.0/(x1*x2**3*x3**2),
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T
	return val, grad

def golinski_constraint3(x, return_grad = False):
	"""Third constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - 1.93/(x2*x3*x6**4)*x4**3
	if not return_grad: return val
	grad = np.vstack([
			0*x1,
			1.93*x4**3/(x2**2*x3*x6**4),
			-5.79*x4**2/(x2*x3*x6**4),
			0*x1,
			7.72*x4**3/(x2*x3*x6**5),
			0*x1,
		]).T
	return val, grad
	

def golinski_constraint4(x, return_grad = False):
	"""Fourth constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - 1.93/(x2*x3*x7**4)*x5**3
	if not return_grad: return val
	grad = np.vstack([
			0*x1,
			1.93*x5**3/(x2**2*x3*x7**4),
			0*x1,
			-5.79*x5**2/(x2*x3*x7**4),
			0*x1,
			7.72*x5**3/(x2*x3*x7**5),	
		]).T
	return val, grad
	

def golinski_constraint5(x, return_grad = False):
	"""Fifth constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - np.sqrt( (745*x4/x2/x3)**2 + 16.9e6)/(110.0*x6**3)
	if not return_grad: return val
	grad = np.vstack([
			0*x1,
			5045.68181818182*x4**2/(x2**3*x3**2*x6**3*np.sqrt(16900000.0 + 555025*x4**2/(x2**2*x3**2))),
			-5045.68181818182*x4/(x2**2*x3**2*x6**3*np.sqrt(16900000.0 + 555025*x4**2/(x2**2*x3**2))),
			0*x1,
			0.0272727272727273*np.sqrt(16900000.0 + 555025*x4**2/(x2**2*x3**2))/x6**4,
			0*x1,
		]).T
	return val, grad

def golinski_constraint6(x, return_grad = False):
	"""Sixth constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - np.sqrt( (745*x5/x2/x3)**2 + 157.5e6)/(85.0*x7**3)
	if not return_grad: return val
	grad = np.vstack([
			0*x1, 
			6529.70588235294*x5**2/(x2**3*x3**2*x7**3*np.sqrt(157500000.0 + 555025*x5**2/(x2**2*x3**2))),
			0*x1, 
			-6529.70588235294*x5/(x2**2*x3**2*x7**3*np.sqrt(157500000.0 + 555025*x5**2/(x2**2*x3**2))),
			0*x1,
			0.0352941176470588*np.sqrt(157500000.0 + 555025*x5**2/(x2**2*x3**2))/x7**4,
		]).T
	return val, grad

def golinski_constraint7(x, return_grad = False):
	"""Seventh constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - x2*x3/40
	if not return_grad: return val
	grad = np.vstack([
			0*x1,
			-x3/40,
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T
	return val, grad

def golinski_constraint8(x, return_grad = False):
	"""Eight constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - 5*x2/x1
	if not return_grad: return val
	grad = np.vstack([
			5*x2/x1**2,
			-5/x1,
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T
	return val, grad

def golinski_constraint9(x, return_grad = False):
	"""Ninth constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - x1/(12*x2)
	if not return_grad: return val
	grad = np.vstack([
			-1/(12*x2),
			x1/(12*x2**2),
			0*x1,
			0*x1,
			0*x1,
			0*x1,
		]).T
	return val, grad

def golinski_constraint24(x, return_grad = False):
	"""Twenty-fourth constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - (1.5*x6 + 1.9)/x4 
	if not return_grad: return val
	grad = np.vstack([
			0*x1,
			0*x1,
			(1.5*x6 + 1.9)/x4**2,
			0*x1,
			-1.5/x4,
			0*x1,
		]).T
	return val, grad
	

def golinski_constraint25(x, return_grad = False):
	"""Twenty-fifth constraint from the Golinski Gearbox problem

	This constraint is satisfied if its value is non-negative.
	"""
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	val = 1 - (1.1*x7 + 1.9)/x5
	if not return_grad: return val
	grad = np.vstack([
			0*x1,
			0*x1,
			0*x1,
			(1.1*x7 + 1.9)/x5**2,
			0*x1,
			-1.1/x5,
		]).T
	return val, grad

# Units of cm
def build_golinski_design_domain():
	return BoxDomain([2.6,0.7, 7.3, 7.3, 2.9, 5.0], [3.6, 0.8, 8.3, 8.3, 3.9, 5.5])
# Taken from table 3 of Hu, Zhou, Chen, Parks, 2017 in AIAA journal 
def build_golinski_random_domain(clip = None):
	return ComboDomain([NormalDomain(0,21e-4**2, clip = clip),
				NormalDomain(0, 1e-4**2, clip = clip),
				NormalDomain(0, 30e-4**2, clip = clip),
				NormalDomain(0, 30e-4**2, clip = clip),
				NormalDomain(0, 21e-4**2, clip = clip),
				NormalDomain(0, 30e-4**2, clip = clip)])
	#return NormalDomain(np.zeros((6,1)), np.diag(np.array([21e-4, 1e-4, 30e-4, 30e-4,21e-4, 30e-4])**2), clip = clip)

if __name__ == '__main__':
	from ridge import PolynomialRidgeApproximation
	import matplotlib.pyplot as plt

	dom = build_golinski_design_domain()
	
	X = dom.sample(draw = 100)
	fX = np.array([golinski_volume(x) for x in X])
	X_norm = dom.normalize(X)
		
	pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, n_init = 10)
	pra.fit(X, fX)
	pra.plot()
	plt.show()

