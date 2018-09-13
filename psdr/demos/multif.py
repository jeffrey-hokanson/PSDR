""" Code for interfacing with MULTI-F:

https://github.com/vmenier/MULTIF/tree/feature_elliptical

This borrows code by Rick Fenrich 
"""
from __future__ import print_function
import numpy as np
from _multif_domains3d import buildDesignDomain, buildRandomDomain

def build_multif_domain(clip = None):
	design_domain = buildDesignDomain(output = 'none')
	random_domain = buildRandomDomain(clip = clip)
	return design_domain + random_domain

def build_multif_design_domain(output = 'none'):
	return buildDesignDomain(output = output)

def build_multif_random_domain(clip = None):
	return buildRandomDomain(clip = clip)


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
	import shutil, subprocess, os, tempfile, shlex
	import numpy as np
	from subprocess import Popen, PIPE, STDOUT

	if workdir is None:
		workdir = tempfile.mkdtemp()
		assert keep_data == False, "In order to keep the run, specify a path for a directory"
	else:
		workdir = os.path.abspath(workdir)
		os.makedirs(workdir)
		
	# Copy the configuration file	
	dir_path = os.path.dirname(os.path.realpath(__file__))
	shutil.copyfile('%s/general-3d.cfg.%s' % (dir_path, version,), workdir + '/general-3d.cfg')
	
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


