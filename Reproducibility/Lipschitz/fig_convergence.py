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

# Sample to build "true" Lipschitz matrix
xs = [np.linspace(lb, ub, 4) for lb, ub in zip(fun.domain.lb, fun.domain.ub)]
Xs = np.meshgrid(*xs, indexing = 'ij')
Xcorner = np.vstack([X.flatten() for X in Xs]).T

fXcorner = fun(Xcorner)
gradXcorner = fun.grad(Xcorner) 

print("#### Identiftying 'true' Lipschitz matrix ####")
lipschitz = LipschitzMatrix(verbose = True)
lipschitz.fit(grads = gradXcorner) 

Ltrue = np.copy(lipschitz.L)
Htrue = np.copy(lipschitz.H)

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
	return np.sqrt(np.sum( ( np.log(a) - np.log(b) )**2 ) )


# Build random realizations for different data
M = 200
N = 1000
reps = 100
mismatch_samp = np.zeros((reps,M+1))
mismatch_grad = np.zeros((reps,N+1))

time_samp = np.zeros(mismatch_samp.shape)
time_grad = np.zeros(mismatch_grad.shape)

lipschitz = LipschitzMatrix(verbose = False)

for i in range(mismatch_samp.shape[0]):
	np.random.seed(i)
	X = fun.domain.sample(max(mismatch_samp.shape[1], mismatch_grad.shape[1]))
	fX = fun(X)
	grads = fun.grad(X)
	
#	for N in tqdm(range(2,mismatch_samp.shape[1]), desc = 'sample   %3d' % i):
#		start_time = time.clock()
#		lipschitz.fit(X[:N], fX[:N])
#		stop_time = time.clock()
#		time_samp[i,N] = stop_time - start_time
#
#		Hsamp = np.copy(lipschitz.H)
#		mismatch_samp[i,N] = metric(Hsamp)
		
	for N in tqdm(range(1,mismatch_grad.shape[1], 100), desc = 'gradient %3d' % i):
		start_time = time.clock()
		lipschitz.fit(grads = grads[:N])
		stop_time = time.clock()
		Hsamp = np.copy(lipschitz.H)
		time_grad[i,N] = stop_time - start_time
		mismatch_grad[i,N] = metric(Hsamp)
		print(mismatch_grad[i,N])

	
	# Now export the data to PGF
	pgf = PGF()
	M = np.arange(2,mismatch_samp.shape[1])
	pgf.add('M', M)
	p0, p25, p50, p75, p100 = np.percentile(mismatch_samp[:i+1,2:], [0, 25, 50, 75, 100], axis =0)
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_samp.dat')
	
	# Now the time part
	pgf = PGF()
	M = np.arange(2,time_samp.shape[1])
	pgf.add('M', M)
	p0, p25, p50, p75, p100 = np.percentile(time_samp[:i+1,2:], [0, 25, 50, 75, 100], axis =0)
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_time_samp.dat')
	
	pgf = PGF()
	M = np.arange(1,mismatch_grad.shape[1])
	pgf.add('M', M)
	p0, p25, p50, p75, p100 = np.percentile(mismatch_grad[:i+1,1:], [0, 25, 50, 75, 100], axis =0)
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_grad.dat')
	
	pgf = PGF()
	M = np.arange(1,time_grad.shape[1])
	pgf.add('M', M)
	p0, p25, p50, p75, p100 = np.percentile(time_grad[:i+1,1:], [0, 25, 50, 75, 100], axis =0)
	pgf.add('p0', p0)
	pgf.add('p25', p25)
	pgf.add('p50', p50)
	pgf.add('p75', p75)
	pgf.add('p100', p100)
	pgf.write('data/fig_convergence_time_grad.dat')

