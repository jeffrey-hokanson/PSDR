""" Code for interfacing with MULTI-F:

https://github.com/vmenier/MULTIF/tree/feature_elliptical

This borrows code by Rick Fenrich 
"""
from __future__ import print_function
import numpy as np

from ._multif_domains3d import buildDesignDomain, buildRandomDomain

def build_multif_domain(truncate = 1e-7):
	design_domain = buildDesignDomain(output = 'none')
	random_domain = buildRandomDomain(truncate = truncate)
	return design_domain * random_domain

def build_multif_design_domain(output = 'none'):
	return buildDesignDomain(output = output)

def build_multif_random_domain(truncate = 1e-7):
	return buildRandomDomain(truncate = truncate)


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
	call = "docker run -t --rm --mount type=bind,source='%s',target='/workdir' jeffreyhokanson/multif:%s" % (workdir, version)
	call += " -f general-3d.cfg -l %d " % (level,)
	if cores is not None:
		# This seems to work on Linux
		#raise NotImplementedError("Haven't figured out how to run docker in parallel")
		call += ' -c %d ' % (cores,)

	# In order to run from inside jupyter, we need to call using Popen
	# following https://github.com/takluyver/rt2-workshop-jupyter/blob/e7fde6565e28adf31a0f9003094db70c3766bd6d/Subprocess%20output.ipynb
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

input_names = [' ']*136

input_names[-40:] = [
	'CMC Density', 
	'CMC Elastic Modulus',
	'CMC Poisson Ratio',
	'CMC Thermal Conductivity',
	'CMC Thermal Expansion Coef',
	'CMC Principle Failure Strain',
	'CMC Max Service Temperature',
	'GR-BMI Density',
	'GR-BMI Elastic Modulus 1',
	'GR-BMI Elastic Modulus 2',
	'GR-BMI Shear Modulus',
	'GR-BMI Poisson Ratio',
	'GR-BMI Mutual Influence Coef 1',
	'GR-BMI Mutual Influence Coef 2',
	'GR-BMI Thermal Conductivity 1',
	'GR-BMI Thermal Conductivity 2',
	'GR-BMI Thermal Conductivity 3',
	'GR-BMI Thermal Expansion Coef 1',
	'GR-BMI Thermal Expansion Coef 2',
	'GR-BMI Thermal Expansion Coef 3',
	'GR-BMI Local Failure Strain 1',
	'GR-BMI Local Failure Strain 2',
	'GR-BMI Local Failure Strain 3',
	'GR-BMI Local Failure Strain 4',
	'GR-BMI Local Failure Strain 5',
	'GR-BMI Max Service Temperature',
	'TI-HC Density',
	'TI-HC Elastic Modulus',
	'TI-HC Poisson Ratio',
	'TI-HC Thermal Conductivity',
	'TI-HC Thermal Expansion Coef',
	'TI-HC Yield Stress',
	'TI-HC Max Service Temperature',
	'Air Thermal Conductivity',
	'Panel Yield Stress',
	'Inlet PSTAG',
	'Inlet TSTAG',
	'ATM Pressure',
	'ATM Temperature',
	'Heat Transfer Coef to Env.',
	]		
