import numpy as np
from .. import BoxDomain, NormalDomain, ComboDomain

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
	x3 = 17 # This is the troublesome integer value
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


def golinski_volume(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)

	val = 0.7854*x1*x2**2*( 3.3333*x3**2 + 14.9334*x3 - 43.0934) \
		-1.5079*x1*(x6**2 + x7**2) + 7.477*(x6**3 + x7**3) + 0.7854*(x4*x6**2 + x5*x7**2)

	return val

# All constraints are satisfied if below 1
# These are taken from Ray 2003, AIAA Journal
def golinski_constraint1(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 27/(x1*x2**2*x3) - 1

def golinski_constraint2(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 397.5/(x1*x2**2*x3**2) - 1

def golinski_constraint3(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 1.93/(x2*x3*x6**4)*x4**3 - 1

def golinski_constraint4(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 1.93/(x2*x3*x7**4)*x5**3 - 1

def golinski_constraint5(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.sqrt( (745*x4/x2/x3)**2 + 16.9e6)/(110.0*x6**3) - 1

def golinski_constraint6(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return np.sqrt( (745*x5/x2/x3)**2 + 157.5e6)/(85.0*x7**3) - 1

def golinski_constraint7(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return x2*x3/40 - 1

def golinski_constraint8(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return 5*x2/x1 - 1

def golinski_constraint9(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return x1/(12*x2) - 1

def golinski_constraint24(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return (1.5*x6 + 1.9)/x4 - 1

def golinski_constraint25(x):
	x1, x2, x3, x4, x5, x6, x7 = expand_variables(x)
	return (1.1*x7 + 1.9)/x5 - 1

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

