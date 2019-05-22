from __future__ import print_function
import numpy as np

from psdr import Function, BoxDomain

class NACA0012(Function):
	r""" A test problem from OpenAeroStruct

	A test problem using OpenAeroStruct similar to that described in [JHM18]_.

	References
	----------
	.. [JHM18] Open-source coupled aerostructural optimization using Python.
		John P. Jasa, John T. Hwang, and Joaquim R. R. A. Martins.
		Structural and Multidisciplinary Optimization (2018) 57:1815-1827
		DOI: 10.1007/s00158-018-1912-8
	
	"""
	def __init__(self, n_lower = 10, n_upper = 10, fraction = 0.1):
		domain = build_hicks_henne_domain(n_lower, n_upper, fraction = fraction)
		Function.__init__(self, naca0012_func, domain, vectorized = False)

def build_hicks_henne_domain(n_lower = 10, n_upper = 10, fraction = 0.1):
	dom = BoxDomain(-fraction*np.ones(n_lower), fraction*np.ones(n_lower), names = 'lower bump') * \
		BoxDomain(-fraction*np.ones(n_upper), fraction*np.ones(n_upper), names = 'upper bump')

	return dom

def naca0012_func(x, version = 'v1', workdir = None, verbose = False, keep_data = False):
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
	else:
		workdir = os.path.abspath(workdir)
		os.makedirs(workdir)

	# Copy the inputs to a file
	np.savetxt(workdir + '/my.input', x, fmt = '%.15e')
	
	call = "docker run -t --rm --mount  type=bind,source='%s',target='/workdir' jeffreyhokanson/naca0012:%s /workdir/my.input" % (workdir, version)
	args = shlex.split(call)
	with open(workdir + '/output.log', 'a') as log:
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

	Y = np.loadtxt(workdir + '/my.output')

	if not keep_data:
		shutil.rmtree(workdir) 
	return Y	
	

if __name__ == '__main__':
	naca = NACA0012()
	X = naca.domain.sample(10)
	print(naca.domain_app.names)
	#Y = naca(X, verbose = True)	
	#print(Y)
