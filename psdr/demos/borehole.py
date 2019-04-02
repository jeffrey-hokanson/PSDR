import numpy as np

from psdr import BoxDomain, NormalDomain, Function


__all__ = ['build_borehole_domain', 'borehole', 'Borehole']

# TODO: implment the random domain version of the input domain


class Borehole(Function):
	r""" The borehole test function
	"""

	def __init__(self, dask_client = None):
		domain = build_borehole_domain()
		funs = [borehole]
		grads = [borehole_grad]

		Function.__init__(self, funs, domain, grads = grads, vectorized = True, dask_client = dask_client)

def build_borehole_domain():
	# Parameters
	# r_w, r, T_u, H_u, T_l, H_l, L, K_w
	lb = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
	ub = np.array([0.15, 50e3, 115600, 1110, 116, 820, 1680, 12045])

	return BoxDomain(lb, ub)


def borehole(X):
	""" Borehole test function

	See: https://www.sfu.ca/~ssurjano/borehole.html
	"""
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
