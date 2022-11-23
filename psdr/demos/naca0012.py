from __future__ import print_function
import numpy as np

from psdr import Function, BoxDomain

__all__ = ['NACA0012']

class NACA0012(Function):
	r""" The lift and drag of a NACA0012 airfoil perturbed by Hicks-Henne bump functions

	This function computes the lift and drag of a NACA0012 airfoil perturbed by Hicks-Henne
	bump functions.  The lift and drag are computed using [SU2]_ using Euler flow at Mach number
	0.8 and angle of attack 1.25. 

	This example slightly modifies the configuration file that appears in the SU2 quickstart folder:
	(inv_NACA0012.cfg)[https://github.com/su2code/SU2/blob/master/QuickStart/inv_NACA0012.cfg].

	Parameters
	----------
	n_lower: int, optional (default: 10)
		Number of bump functions for the lower surface
	n_upper: int, optional (default: 10)
		Number of bump functions for the upper surface
	fraction: float, optional (default:0.01)
		 

	References
	----------
	.. [SU2] https://su2code.github.io/	
	"""
	def __init__(self, n_lower = 10, n_upper = 10, fraction = 0.01, dask_client = None, **kwargs):
		domain = build_hicks_henne_domain(n_lower, n_upper, fraction = fraction)
		kwargs.update({'n_lower':n_lower, 'n_upper':n_upper}) 
		Function.__init__(self, naca0012_func, domain, vectorized = False,
			kwargs = kwargs, dask_client = dask_client, return_grad = True)


def build_hicks_henne_domain(n_lower = 10, n_upper = 10, fraction = 0.1):
	dom = BoxDomain(-fraction*np.ones(n_lower), fraction*np.ones(n_lower), names = 'lower bump') * \
		BoxDomain(-fraction*np.ones(n_upper), fraction*np.ones(n_upper), names = 'upper bump')

	return dom


def naca0012_func(x, version = 'v1', workdir = None, verbose = False, 
	keep_data = False, n_lower = 10, n_upper = 10, return_grad = False, maxiter = 1000,
	nprocesses = 1):
	r"""



	"""
	# If we use this inside the RedisPool, we need to load the modules
	# internal to this file
	import shutil, subprocess, os, tempfile, shlex, platform
	import numpy as np
	from subprocess import Popen, PIPE, STDOUT

	if workdir is None:
		# Docker cannot access /var by default, so we move the temporary file to
		# /tmp on MacOS
		if platform.system() == 'Darwin':
			workdir = tempfile.mkdtemp(dir = '/tmp')
		else:
			workdir = tempfile.mkdtemp()
		assert keep_data == False, "In order to keep the run, specify a path for a directory"
	else:
		workdir = os.path.abspath(workdir)
		os.makedirs(workdir)

	# Copy the inputs to a file
	np.savetxt(workdir + '/my.input', x, fmt = '%.15e')
	
	call = "docker run -t --rm --mount  type=bind,source='%s',target='/workdir' jeffreyhokanson/naca0012:%s /workdir/my.input" % (workdir, version)
	call += " --nlower %d --nupper %d" % (n_lower, n_upper)
	if return_grad:
		call += " --adjoint discrete"
	call += ' --maxiter %d' % maxiter
	call += ' --nprocesses %d' % nprocesses
	args = shlex.split(call)
	with open(workdir + '/output.log', 'ab') as log:
		p = Popen(args, stdout = PIPE, stderr = STDOUT)
		while True:
			# Read output from pipe
			# TODO: this should buffer to end of line rather than fixed size
			output = p.stdout.readline()
			log.write(output)

			if verbose:
				print(output, end ='')

			# Check for termination
			if p.poll() is not None:
				break
		if p.returncode != 0:
			print("exited with error code %d" % p.returncode)

	fx = np.loadtxt(workdir + '/my.output')

	if return_grad:
		grad = np.loadtxt(workdir + '/my_grad.output')

	if not keep_data:
		shutil.rmtree(workdir) 
	
	if return_grad:
		return fx, grad
	else:
		return fx
	

