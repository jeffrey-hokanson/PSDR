import numpy as np

def check_gradient(f, x, grad, verbose = False):
	grad_est_best = np.zeros(grad.shape)
	for h in np.logspace(-14, -2,10):
		grad_est = np.zeros(grad.shape)
		for i in np.ndindex(x.shape):
			# Build unit vector
			ei = np.zeros(x.shape)
			ei[i] = 1
			# Construct finite difference approximation
			d = (f(x + h*ei) - f(x - h*ei))/(2*h)
			# Add to gradient approximation
			idx = tuple([slice(0,ni) for ni in d.shape] + list(i))
			grad_est[idx] = (f(x + h*ei) - f(x - h*ei))/(2*h)
		if np.max(np.abs(grad - grad_est)) < np.max(np.abs(grad - grad_est_best)):
			grad_est_best = grad_est

	if verbose:
		for i in np.ndindex(grad.shape):
			print i, "%+5.2e %+5.2e  %+5.2e" %(grad[i], grad_est_best[i], grad[i]/grad_est_best[i])

	return np.max(np.abs(grad - grad_est_best))
