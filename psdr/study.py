# Convergence studies
import numpy as np
from scipy.linalg import subspace_angles

def subspace_convergence(sdr, fun, sampler, Ns,  domain = None, data = 'eval', subspace_dimension = 1):
	r""" Measure the convergence of a subspace-based dimension reduction technique

	This is a convience wrapper 

	Parameters
	----------
	sdr: instance of SubspaceBasedDimensionReduction
		technique for performing subspace based dimension reduction
	fun: instance of Function
		Function to perform the analysis on 
	sampler: function

	domain: (optional), defaults to fun.domain
		Domain which to sample
	data: ['eval', 'grad']
		What data to sample
	subspace_dimension: int
		Number of subspace dimensions to compare

	Returns
	-------
	subspace_angle: np.array
		List of largest subspace angle between sequential subspaces
	N: np.array
		Number of queries used to construct subspace approximation
	"""
	if domain is None:
		domain = fun.domain

	assert data in ['eval', 'grad'], "Data must be one of either eval or gradient"

	Us = []
	Ms = []
	for N in Ns:
		# Evaluate sampling scheme
		X = sampler(domain, N).reshape(-1, len(domain))
		# Compute the number of points actually sampled
		Ms.append(len(X))

		# Perform subspace estimation
		if data == 'eval':
			fX = fun(X)
			sdr.fit(X = X, fX = fX)
		elif data == 'grad':
			grads = fun.grad(X)
			sdr.fit(grads = grads)

		Us.append(np.copy(sdr.U[:,:subspace_dimension]))

	# Compute angles
	angles = [np.pi/2] + [np.max(subspace_angles(Us[i], Us[i+1])) for i in range(len(Us)-1)]
	return np.array(angles), np.array(Ms)
		
