from __future__ import print_function
import numpy as np

from psdr import BoxDomain, TensorProductDomain, UniformDomain, Function



class OpenAeroStruct(Function):
	r""" A test problem from OpenAeroStruct

	A test problem using OpenAeroStruct similar to that described in [JHM18]_.

	References
	----------
	.. [JHM18] Open-source coupled aerostructural optimization using Python.
		John P. Jasa, John T. Hwang, and Joaquim R. R. A. Martins.
		Structural and Multidisciplinary Optimization (2018) 57:1815-1827
		DOI: 10.1007/s00158-018-1912-8
	
	"""
	def __init__(self):
		domain = build_oas_design_domain() * build_oas_robust_domain() * build_oas_random_domain()
		Function.__init__(self, oas_func, domain, vectorized = True)

	def __str__(self):
		return "<Open Aero Struct Function>"

def build_oas_design_domain(n_cp = 3):
	# Twist
	domain_twist = BoxDomain(-1*np.ones(n_cp), 1*np.ones(n_cp), names = ['twist %d' % (i,) for i in range(1,4)] )
	# Thick
	domain_thick = BoxDomain(0.005*np.ones(n_cp), 0.05*np.ones(n_cp), names = ['thickness %d' % (i,) for i in range(1,4)] )
	# Root Chord
	domain_root_chord = BoxDomain(0.7, 1.3, names = ['root chord'])
	# Taper ratio
	domain_taper_ratio = BoxDomain(0.75, 1.25, names = ['taper ratio'])

	return TensorProductDomain([domain_twist, domain_thick, domain_root_chord, domain_taper_ratio])

def build_oas_robust_domain():
	# alpha - Angle of Attack
	return BoxDomain(2.0, 5.0, names = ['angle of attack'])

def build_oas_random_domain():
	E = UniformDomain(0.8*70e9, 1.2*70e9, names = ["Young's modulus of the spar [Pa]"])	 
	G = UniformDomain(0.8*30e9, 1.2*30e9, names = ["sheear modulus of the spar [Pa]"])
	rho = UniformDomain(0.8*3e3, 1.2*3e3, names = ["material density [kg/m^3]"] )
	return TensorProductDomain([E,G,rho])


def oas_func(x, version = 'v1', workdir = None, verbose = False, keep_data = False):
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
	call = "docker run -t --rm --mount type=bind,source='%s',target='/workdir' jeffreyhokanson/oas:%s /workdir/my.input" % (workdir, version)
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

	Y = np.loadtxt(workdir + '/my.output')

	if not keep_data:
		shutil.rmtree(workdir) 
	return Y	
	


