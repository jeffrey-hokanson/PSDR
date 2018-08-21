import numpy as np
from .. import BoxDomain

__all__ = ['build_borehole_domain', 'borehole']

# TODO: implment the random domain version of the input domain


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

	f = 2*np.pi*T_u*(H_u - H_l)/(np.log(r/r_w)*(1 + 2*L*T_u/(np.log(r/r_w)*r_w**2*K_w) + T_u/T_l))
	return f
