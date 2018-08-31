import numpy as np
import matplotlib.pyplot as plt

def fdcheck(f, fp, dof, gen_fig=0, nsamp=10, x=None, xp=None, ord=2, xmin=-8, xmax=0):
	"""custome finite difference checker

	Parameters
	----------
	f : callable (one arg)
		function to compute finite difference
	fp : callable (two arg, x and dx, respectively)
		derivative of f (reference for error)
	dof : int
		degrees of freedom of both f and fp
	gen_fig: bool
		boolean flag to display finite difference V shape

	Returns
	-------
	numpy.float64
		minimum error in the finite difference V
	"""
	if x is None:
		x = np.random.randn(dof)
	if xp is None:
		xp = np.random.randn(dof)
	error = np.zeros((nsamp,))
	if nsamp == 1:
		eps_list = np.atleast_1d(1e-3)
	else:
		eps_list = np.logspace(xmin, xmax, nsamp)
	df = fp(x, xp)
	for i, eps in enumerate(eps_list):
		fp1 = f(x + eps * xp)
		fm1 = f(x - eps * xp)
		if ord==1:
			df_fd = (0.5 * fp1 - 0.5 * fm1) / (eps)
		else:
			fp2 = f(x + 2 * eps * xp)
			fm2 = f(x - 2 * eps * xp)
			df_fd = (- 1./12 * fp2 + 2./3 * fp1 - 2./3 * fm1 + 1./12 * fm2) / (eps)
		if np.linalg.norm(df) < 1e-15:
			error[i] = np.linalg.norm(df_fd - df)
		else:
			error[i] = np.linalg.norm(df_fd - df) / np.linalg.norm(df)
	if gen_fig:
		plt.loglog(eps_list, error)
		plt.show()
	return min(error)

def fdcheck2(f, fpp, dof, gen_fig=0, nsamp=10, x=None, xpp=None, ord=1, xmin=-8, xmax=0):
	if x is None:
		x = np.random.randn(dof)
	if xpp is None:
		xpp = np.random.randn(dof)
	error = np.zeros((nsamp,))
	if nsamp == 1:
		eps_list = np.atleast_1d(1e-2)
	else:
		eps_list = np.logspace(xmin, xmax, nsamp)
	ddf = fpp(x, xpp)
	# print 'ddf = %1.2e' % np.linalg.norm(ddf)
	for i, eps in enumerate(eps_list):
		fp1 = f(x + eps * xpp)
		f0 = f(x)
		fm1 = f(x - eps * xpp)
		if ord==1:
			ddf_fd = (fp1 - 2 * f0 + fm1) / (eps**2)
		else:
			fp2 = f(x + 2 * eps * xpp)
			fm2 = f(x - 2 * eps * xpp)
			ddf_fd = (- 1./12 * fp2 + 4./3 * fp1 - 5./2 * f0 + 4./3 * fm1 - 1./12 * fm2) / (eps**2)
		if np.linalg.norm(ddf) < 1e-15:
			error[i] = np.linalg.norm(ddf_fd - ddf)
		else:
			error[i] = np.linalg.norm(ddf_fd - ddf) / np.linalg.norm(ddf)
		# print 'ddf_fd = %1.2e' % np.linalg.norm(ddf_fd)
	if gen_fig:
		plt.loglog(eps_list, error)
		plt.show()
	return min(error)
