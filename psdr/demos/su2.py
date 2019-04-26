from __future__ import print_function
import numpy as np

from psdr import Function, BoxDomain



class SU2(Function):
	r""" A test problem from OpenAeroStruct

	A test problem using OpenAeroStruct similar to that described in [JHM18]_.

	References
	----------
	.. [JHM18] Open-source coupled aerostructural optimization using Python.
		John P. Jasa, John T. Hwang, and Joaquim R. R. A. Martins.
		Structural and Multidisciplinary Optimization (2018) 57:1815-1827
		DOI: 10.1007/s00158-018-1912-8
	
	"""
	def __init__(self, n_lower = 10, n_upper = 10):
		domain = build_hicks_henne_domain(n_lower, n_upper)
		# build_oas_design_domain() * build_oas_robust_domain() * build_oas_random_domain()
		Function.__init__(self, oas_func, domain, vectorized = True)

def build_hicks_henne_domain(n_lower = 10, n_upper = 10):
	dom = BoxDomain(-0.1*np.ones(n_lower), 0.1*np.ones(n_lower), names = 'lower bump') * \
		BoxDomain(-0.1*np.ones(n_upper), 0.1*np.ones(n_upper), names = 'upper bump')


def su2_func(x, version = 'v1', workdir = None, verbose = False):
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
	
	call = "docker run -t --rm --mount  type=bind,source='%s',target='/workdir' laksharma30/naca0012:%s /workdir/my.input" % (workdir, version)
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

	#shutil.rmtree(workdir) 
	return Y	
	



if __name__ == '__main__':
	su2 = SU2()
	X = su2.sample(10)
	print(su2.domain_app.names)
	Y = su2(X)	
	print(Y)
