from __future__ import print_function
import numpy as np
from psdr.demos import OTLCircuit
from psdr.pgf import PGF
from psdr import LipschitzMatrix
import scipy.linalg
from tqdm import tqdm
import time

# Build the domain and function
fun = OTLCircuit()

try:
	Htrue = np.loadtxt('data/fig_convergence_Htrue.dat')
except IOError:
	
	# Sample to build "true" Lipschitz matrix
	xs = [np.linspace(lb, ub, 6) for lb, ub in zip(fun.domain.lb, fun.domain.ub)]
	Xs = np.meshgrid(*xs, indexing = 'ij')
	Xcorner = np.vstack([X.flatten() for X in Xs]).T

	fXcorner = fun(Xcorner)
	gradXcorner = fun.grad(Xcorner) 

	print("#### Identiftying 'true' Lipschitz matrix ####")
	lipschitz = LipschitzMatrix(verbose = True)
	lipschitz.fit(grads = gradXcorner) 
	Htrue = np.copy(lipschitz.H)
	np.savetxt('data/fig_convergence_Htrue.dat', Htrue)

def metric(Hsamp):
	""" Distance metric from BS09: eq 2.5:
		
		delta_2 = sum_i log^2( lambda_i(A, B))

	where lambda_i(A, B) are the generalized eigenvalues of A, B
	"""

	# TODO: Should we use the homogenous form (alpha, beta) in Stewart and Sun),
	# which avoids numerical issues for large/small eigenvales 
	# or switch to eigh which cannot return the homogenous form
	a, b = scipy.linalg.eig(Htrue, Hsamp, homogeneous_eigvals = True, left = False, right = False)
	a = a.real
	b = b.real
	d2 = np.sqrt(np.sum( ( np.log(a) - np.log(b) )**2 ) )
	if not np.isfinite(d2):
		d2 = np.inf
	return d2


# Build random realizations for different data
M = 200
N = 1000

#Mvec = np.linspace(2,200)
Mvec = np.unique(np.logspace(np.log10(2), np.log10(500), 50).astype(np.int))
Nvec = np.unique(np.logspace(np.log10(1), np.log10(1e4), 50).astype(np.int))

reps = 100
mismatch_samp = np.zeros((reps,len(Mvec)))
mismatch_grad = np.zeros((reps,len(Nvec)))


time_samp = np.zeros(mismatch_samp.shape)
time_grad = np.zeros(mismatch_grad.shape)

lipschitz = LipschitzMatrix(verbose = True)

for rep in range(mismatch_samp.shape[0]):
	np.random.seed(rep)
	X = fun.domain.sample(max(np.max(Mvec), np.max(Nvec)))
	fX = fun(X)
	grads = fun.grad(X)
	
		
	#for i, N in tqdm(enumerate(Nvec), desc = 'gradient %3d' % rep, total = len(Nvec)):
	for i, N in enumerate(Nvec):
		print("\n\n#### grads  N = %d, rep %d" % (N,rep))
		start_time = time.clock()
		lipschitz.fit(grads = grads[:N])
		stop_time = time.clock()
		time_grad[rep, i] = stop_time - start_time
		Hsamp = np.copy(lipschitz.H)
		mismatch_grad[rep, i] = metric(Hsamp)
		print("mismatch: ", mismatch_grad[rep,i])	

	pgf = PGF()
	pgf.add('N', Nvec)
	# Nearest interpolation removes nans if some values are infinite
	p0, p25, p50, p75, p100 = np.percentile(mismatch_grad[:rep+1], [0, 25, 50, 75, 100], axis =0, 
		interpolation = 'nearest')
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_grad.dat')
	
	pgf = PGF()
	pgf.add('N', Nvec)
	p0, p25, p50, p75, p100 = np.percentile(time_grad[:rep+1], [0, 25, 50, 75, 100], axis =0)
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_time_grad.dat')


	#for i, M in tqdm(enumerate(Mvec), desc = 'sample   %3d' % rep, total = len(Mvec)):
	for i, M in enumerate(Mvec):
		print("\n\n#### samples  M = %d, rep %d" % (M,rep))
		start_time = time.clock()
		lipschitz.fit(X[:M], fX[:M])
		stop_time = time.clock()
		time_samp[rep, i] = stop_time - start_time
		Hsamp = np.copy(lipschitz.H)
		mismatch_samp[rep, i] = metric(Hsamp)
		print("mismatch: ", mismatch_samp[rep,i])	
	
	# Now export the data to PGF
	pgf = PGF()
	pgf.add('M', Mvec)
	p0, p25, p50, p75, p100 = np.percentile(mismatch_samp[:rep+1], [0, 25, 50, 75, 100], axis =0,
		interpolation = 'nearest')
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_samp.dat')
	
	# Now the time part
	pgf = PGF()
	pgf.add('M', Mvec)
	p0, p25, p50, p75, p100 = np.percentile(time_samp[:rep+1], [0, 25, 50, 75, 100], axis =0)
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_time_samp.dat')
	

