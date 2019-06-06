""" Code for interfacing with MULTI-F:

https://github.com/vmenier/MULTIF/tree/feature_elliptical

This borrows code by Rick Fenrich 
"""
from __future__ import print_function
import numpy as np
from psdr import Function
from ._multif_domains3d import buildDesignDomain, buildRandomDomain



class MULTIF(Function):
	r"""An interface to the MULTI-F multiphysics jet nozzle model test problem.

	The function describes a multiphysics model of a jet nozzle [FMAA18]_.
	The domain consists of 96 design parameters and 40 uncertain variables
	and the output returns the mass, thrust, and many different failure constraints.
	This uses a Docker image to encapsulate the dependencies for MULTI-F;
	to use this function, install Docker and pull the multif image


	.. code-block:: bash

	    >> docker pull jeffreyhokanson/multif:v25


	This function encodes multiple different fidelity levels:
	in order of increasing fidelity and approximate one-core cost

	======    =============    =========    ==========    =============
	Level     Physics          Mesh         Mechanics     Runtime (sec)
	======    =============    =========    ==========    =============
	0         1d Non-ideal     N/A          linear        20
	1         1d Non-ideal     N/A          nonlinear     205
	2         2d Euler         Coarse       linear        82
	3         2d Euler         Medium       linear        256
	4         2d Euler         Fine         linear        713
	5         3d Euler         Coarse       linear        1083
	6         3d Euler         Medium       linear        3123
	7         3d Euler         Fine         linear        13350
	8         2d RANS          Coarse       linear        
	9         2d RANS          Medium       linear
	10        2d RANS          Fine         linear
	11        3d RANS          Coarse       linear        80690
	12        3d RANS          Medium       linear        175888
	13        3d RANS          Fine         linear        408512
	======    =============    =========    ==========    =============

	Note at the present time the 2d RANS levels are buggy and not recommended for use.


	Parameters
	----------
	truncate: float in [0,1]; default 1e-7
		If non-zero, truncate the random domains by the specified probability.
		Truncation is necessary to provide a bounded domain for use with most sampling strategies.
	level: int in [0, 13]; default 0
		What level of fidelity to run (see description above)
	su2_maxiter: int; default: 5000
		Number of iterations used to solve for the fluid flow in SU2
	workdir: str; default: None
		If defined, this specifies the location where temporary files are written while solving the problem 
	keep_data: bool; default: False	
		If True, do not delete the work files; i.e., use this to preserve the flow solution for later use
	verbose: bool; default: False
		If True, print the output of running MULTI-F to stdout
	dask_client: dask.distributed.Client or None
		If specified, allows distributed computation with this function.

	References
	----------
	.. [FMAA18] Reliability-Based Design Optimization of a Supersonic Nozzle
		Richard W. Fenrich, Victorien Menier, Philip Avery, and Juan J. Alonso,
		6th European Conference on Computational Mechanics
		http://www.eccm-ecfd2018.org/admin/files/filePaper/p437.pdf

	""" 
	def __init__(self, truncate = 1e-7, level = 0, su2_maxiter = 5000, workdir = None,
		keep_data = False, verbose = False, dask_client = None):
		self.design_domain_app = buildDesignDomain(output = 'none', solver = 'CVXOPT')
		self.design_domain_norm = self.design_domain_app.normalized_domain()
		self.design_domain = self.design_domain_norm		

		self.random_domain_app = buildRandomDomain(truncate = truncate)
		self.random_domain_norm = self.random_domain_app.normalized_domain()
		self.random_domain = self.random_domain_norm		

		domain = self.design_domain_app * self.random_domain_app
		Function.__init__(self, multif, domain, vectorized = False, dask_client = dask_client)

	def __str__(self):
		return "<MULTI-F Function>"

def build_multif_domain(truncate = 1e-7, **kwargs):
	design_domain = buildDesignDomain(output = 'none', **kwargs)
	random_domain = buildRandomDomain(truncate = truncate, **kwargs)
	return design_domain * random_domain

def build_multif_design_domain(output = 'none', **kwargs):
	return buildDesignDomain(output = output, **kwargs)

def build_multif_random_domain(truncate = 1e-7):
	return buildRandomDomain(truncate = truncate, **kwargs)


def multif(x, level = 0, version = 'v25', su2_maxiter = None, workdir = None, 
	keep_data = False, verbose = False, cores = None):
	"""


	*NOTE*:	prior to running, install Docker and then pull the image for multif:

	    >> docker pull jeffreyhokanson/multif:v25


	Parameters
	----------
	x: np.array(136)
		Input coordinates to MULTI-F in the application domain

	level: int
		Level of MULTIF to run, one of 0-13 inclusive

	su2_maxiter: None or int
		Maximum number of iterations to run only for levels 2-13;
		default = 5000

	workdir: string or None
		If None, create a tempory file 
	
	keep_data: bool
		If true, do not delete the directory containing intermediate results

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
		
	# Copy the configuration file	
	dir_path = os.path.dirname(os.path.realpath(__file__))
	shutil.copyfile('%s/multif/general-3d.cfg.%s' % (dir_path, version,), workdir + '/general-3d.cfg')
	
	# If provided a maximum number of SU2 iterations, set that value
	if su2_maxiter is not None:
		with open(workdir + '/general-3d.cfg', 'a') as config:
			config.write("SU2_MAX_ITERATIONS=%d" % su2_maxiter)

	# Copy the input parameters 
	np.savetxt(workdir + '/general-3d.in', x.reshape(-1,1), fmt = '%.15e')
	
	# Now call multif
	uid = os.getuid()
	# We specify the user ID so we can later delete the results by the local user 
	call = 'docker run -t --rm --mount type=bind,source="%s",target="/workdir" --workdir /workdir --user %d' % (workdir, uid)
	call += ' jeffreyhokanson/multif:%s' % (version,)
	call += " -f general-3d.cfg -l %d " % (level,)
	if cores is not None:
		# This seems to work on Linux
		#raise NotImplementedError("Haven't figured out how to run docker in parallel")
		call += ' -c %d ' % (cores,)

	# In order to run from inside jupyter, we need to call using Popen
	# following https://github.com/takluyver/rt2-workshop-jupyter/blob/e7fde6565e28adf31a0f9003094db70c3766bd6d/Subprocess%20output.ipynb
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

	#if verbose:		
	#	return_code = subprocess.call(call, shell = True)
	#else:
	#	with open(workdir + '/output.log', 'a') as output:
	#		return_code = subprocess.call(call, shell = True, stdout = output, stderr = output)

	# Now read output
	with open(workdir + '/results.out') as f:
		output = []
		for line in f:
			value, name = line.split()
			output.append(float(value))
	fx = np.array(output)

	# delete the output if we're not keeping it 
	if not keep_data:
		shutil.rmtree(workdir) 

	return fx


level_names = ["NONIDEALNOZZLE,1e-8,AEROTHERMOSTRUCTURAL,LINEAR,0.01",
		 "NONIDEALNOZZLE,1e-8,AEROTHERMOSTRUCTURAL,NONLINEAR,0.4",
		"EULER,2D,COARSE,AEROTHERMOSTRUCTURAL,LINEAR,0.01",
		"EULER,2D,MEDIUM,AEROTHERMOSTRUCTURAL,LINEAR,0.2",
		"EULER,2D,FINE,AEROTHERMOSTRUCTURAL,LINEAR,0.4",
		"EULER,3D,COARSE,AEROTHERMOSTRUCTURAL,LINEAR,0.01",
		"EULER,3D,MEDIUM,AEROTHERMOSTRUCTURAL,LINEAR,0.2",
		"EULER,3D,FINE,AEROTHERMOSTRUCTURAL,LINEAR,0.4",
		"RANS,2D,COARSE,AEROTHERMOSTRUCTURAL,LINEAR,0.01",
		"RANS,2D,MEDIUM,AEROTHERMOSTRUCTURAL,LINEAR,0.2",
		"RANS,2D,FINE,AEROTHERMOSTRUCTURAL,LINEAR,0.4",
		"RANS,3D,COARSE,AEROTHERMOSTRUCTURAL,LINEAR,0.01",
		"RANS,3D,MEDIUM,AEROTHERMOSTRUCTURAL,LINEAR,0.2",
		"RANS,3D,FINE,AEROTHERMOSTRUCTURAL,LINEAR,0.4"
	]

qoi_names = [
	'SU2 Residual',
	'Mass',
	'Wall Mass', # 2
	'Volume',
	'Thrust', # 4
	'Thermal Layer Temp. Fail. (Max)',
	'Inside Load Layer Temp. Fail. (Max)', #6 
	'Middle Load Layer Temp. Fail. (Max)',
	'Outside Load Layer Temp. Fail. (Max)', #8
	'Thermal Layer Struct. Fail. (Max)', 
	'Inside Load Layer Struct. Fail. (Max)', #10
	'Middle Load Layer Struct. Fail. (Max)', 
	'Outside Load Layer Struct. Fail. (Max)', #12
	'Stringers Struct. Fail. (Max)', 
	'Baffle 1 Struct. Fail. (Max)', # 14
	'Baffle 2 Struct. Fail. (Max)', 
	'Baffle 3 Struct. Fail. (Max)', # 16
	'Baffle 4 Struct. Fail. (Max)', 
	'Baffle 5 Struct. Fail. (Max)', #18
	'Thermal Layer Temp. Fail. (PN)', #19 -----
	'Inside Load Layer Temp. Fail. (PN)', # 20
	'Middle Load Layer Temp. Fail. (PN)', 
	'Outside Load Layer Temp. Fail. (PN)', #22
	'Thermal Layer Struct. Fail. (PN)',
	'Inside Load Layer Struct. Fail. (PN)', #24
	'Middle Load Layer Struct. Fail. (PN)',
	'Outside Load Layer Struct. Fail. (PN)', #26
	'Stringers Struct. Fail. (PN)',
	'Baffle 1 Struct. Fail. (PN)', # 28
	'Baffle 2 Struct. Fail. (PN)',
	'Baffle 3 Struct. Fail. (PN)', # 30
	'Baffle 4 Struct. Fail. (PN)',
	'Baffle 5 Struct. Fail. (PN)', #32
	'Thermal Layer Temp. Fail. (KS)', # 33 --------
	'Inside Load Layer Temp. Fail. (KS)', #34
	'Middle Load Layer Temp. Fail. (KS)',
	'Outside Load Layer Temp. Fail. (KS)', #36
	'Thermal Layer Struct. Fail. (KS)', 
	'Inside Load Layer Struct. Fail. (KS)', #38
	'Middle Load Layer Struct. Fail. (KS)', 
	'Outside Load Layer Struct. Fail. (KS)', # 40
	'Stringers Struct. Fail. (KS)', 
	'Baffle 1 Struct. Fail. (KS)', # 42
	'Baffle 2 Struct. Fail. (KS)', 
	'Baffle 3 Struct. Fail. (KS)', # 44
	'Baffle 4 Struct. Fail. (KS)', 
	'Baffle 5 Struct. Fail. (KS)', # 46
	]

