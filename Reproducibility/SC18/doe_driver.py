import numpy as np
import time

from multifDomains3D import *
from psdr import RedisPool
from psdr import PolynomialRidgeApproximation, stretch_sample 


def fun(x, name, level = 0, version = 'v25', sst = 'false'):
	import shutil, subprocess, os, errno
	import numpy as np
	import time, datetime, rfc3339
	workdir = '/local/scratch/multif_%s' % (name,)
	outdir = '/rc_scratch/jeho8774/multif_%s' % (name,)	

	try:
		os.makedirs(workdir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise e

	try: 
		os.makedirs(outdir)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise e
	
	# Copy the approprieate config
	shutil.copyfile('/home/jeho8774/MULTIF/general-3d.cfg.%s' % (version,), workdir + '/general-3d.cfg')

	# Create the input file
	np.savetxt(workdir + '/general-3d.in', x.reshape(-1,1), fmt = '%.15e')
	np.savetxt(outdir + '/general-3d.in', x.reshape(-1,1), fmt = '%.15e')

	start_time = time.time()
	# Now call the script
	print "calling singularity"
	try:
		# Since this isn't run a shell, it doesn't respect the bashrc PATH and so we specify
		# the exact path to the singularity binary
		return_code = subprocess.call(['/curc/sw/singularity/2.4.2/bin/singularity', 'run', '/projects/jeho8774/multif_%s.img' % version,
			'-f', 'general-3d.cfg', '-l', '%d' % level], cwd = workdir)
	except Exception as e:
		print e
		print "singularity failed"

	print "finished singularity"
	stop_time = time.time()
	
	# copy over the results for safe keeping
	try:
		shutil.copyfile(workdir + '/results.out', outdir + '/results.out')
	except:
		print "output copy failed"
		return None

	# TODO: Record runtime in log
	with open('/home/jeho8774/total_runtimes.txt', 'a') as f:
		# stop time
		timestamp = rfc3339.rfc3339(datetime.datetime.now())
		f.write('%d %s nohash %d 0 %s %s %s\n' % (int(stop_time - start_time), name, level, timestamp, version, sst))

	# extract output
	with open(outdir + '/results.out') as f:
		output = []
		for line in f:
			value, name = line.split()
			output.append(float(value))

	
	y = np.array(output)

	# Delete old data
	shutil.rmtree(workdir)

	return y


if __name__ == '__main__':

	design_domain = buildDesignDomain(output = None)
	random_domain = buildRandomDomain(clip = 2)
	total_domain = design_domain + random_domain
	total_domain_norm = total_domain.normalized_domain()

	pool = RedisPool(name = 'multif')

	prefix = 'stretch'

	level = 11

	X0 = np.loadtxt('rand2_subset.input')
	Y0 = np.loadtxt('rand2_subset_level11_v24.output')
	jobs = []
	i = len(X0)

	qois = [1,4,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]

	max_jobs = 1000

	while len(jobs) < max_jobs:
		nworkers_free = sum([worker.state == 'idle' for worker in pool.workers])
		nsamples = sum([job.state == 'done' for job in jobs]) 
		now = time.strftime("%H:%M:%S")
		print "%s: %d jobs, %d jobs complete, %d workers, %d workers free" % (now, len(jobs), nsamples, len(pool.workers), nworkers_free) 
		if nworkers_free == 0:
			time.sleep(1)
			Xnew = np.zeros((0,len(total_domain)))
		else:
			# If we have enough samples, do an experimental design
			#print "nsamples: %d\t availble workers %d" % (nsamples, nworkers_free)

			# Extract info about current running jobs 
			done_jobs = [job for job in jobs if job.ready()]
			try:
				Xdone = np.vstack([X0] + [job.args[0] for job in done_jobs])
				Ydone = np.vstack([Y0] + [job.output for job in done_jobs])
			except ValueError:
				Xdone = X0
				Ydone = Y0
	
			np.savetxt('%s_inc.input' % prefix, Xdone)
			np.savetxt('%s_inc.output' % prefix, Ydone) 
	
			# Determine which points are already running
			try:
				X = np.vstack([X0] + [job.args[0] for job in jobs])
			except ValueError:
				X = X0

			# Normalize everything
			Xdone_norm = total_domain.normalize(Xdone)
			X_norm = total_domain.normalize(X)

			# Now build ridge approximations
			Us = []
			for qoi in qois:
				pra = PolynomialRidgeApproximation(subspace_dimension = 1, degree = 3)
				I = np.isfinite(Ydone[:,qoi])
				pra.fit(Xdone_norm[I,:], Ydone[I,qoi])
				Us.append(pra.U)

			# Now do the sampling
			Xnew_norm = stretch_sample(total_domain_norm, Us, X0 = X_norm, M = 1, verbose = True, enrich = False)
			print "Added %d new points" % Xnew_norm.shape[0]
	
			Xnew = total_domain.unnormalize(Xnew_norm).reshape(1,-1)


		for x in Xnew:
			i += 1
			#print [job.job_name for job in jobs]
			job_name = '%s_%d' % (prefix, i)
			print "new job name", job_name
			jobs.append(pool.apply_async(fun, args = (x, job_name), kwargs ={'level':level}, job_name = job_name))

	# Now wait for all jobs to be done
	print "Done with DOE.  Waiting for all runs to finish"
	X = np.vstack([X0] + [job.args[0] for job in jobs])
	Y = np.vstack([Y0] + [job.get() for job in jobs])
	
	np.savetxt('%s.input' % prefix, X)
	np.savetxt('%s.output' % prefix, Y)

	print "Finished"	
