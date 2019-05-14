from __future__ import division, print_function
import numpy as np
from random import randint
import psdr
import psdr.demos
from psdr.pgf import PGF
import cvxpy as cp
from tqdm import tqdm
from itertools import product

fun = psdr.demos.OTLCircuit()

# Compute Lipschitz constant/matrix
X = fun.domain.sample_grid(2)
grads = fun.grad(X)

Lcon = psdr.LipschitzConstant()
Lcon.fit(grads = grads)

Lmat = psdr.LipschitzMatrix()
Lmat.fit(grads = grads)

print(Lmat.L)

epsilons = np.logspace(np.log10(1e-4), np.log10(1), 100)[::-1]
#epsilons = np.array([1,1e-1,1e-2,1e-3,1e-4])
Nsamp = int(1e3)
m = len(fun.domain)

def prod(z):
	tot = int(1)
	for zi in z:
		tot *= zi
	return tot


# Rotate onto the principle axes to reduce number of grid points 
U, s, VT = np.linalg.svd(Lmat.L)
Lmat = np.diag(s).dot(VT)
Lcon = Lcon.L
for L, name in zip([Lcon, Lmat], ['con', 'mat']):
#for L, name in zip([Lmat], [ 'mat']):
	# Corners of the transformed domain
	Y = L.dot(X.T).T

	Ns = np.zeros(len(epsilons))
	Nstd = np.zeros(len(epsilons))

	for i, eps in enumerate(epsilons):

		# construct the grid
		grid = []
		grid_spacing = 2*eps/np.sqrt(m)
		for j in range(m):
			ymax = np.max(Y[:,j])
			ymin = np.min(Y[:,j])
			
			if ymax - ymin < 2*grid_spacing:
				# We only need to consider one ball
				grid.append([ (ymin + ymax)/2.])
			else:
				grid.append(np.arange(ymin + grid_spacing, ymax - grid_spacing, grid_spacing ))
	
		# Number of potential grid points
		ngrid = prod([len(grid_axis) for grid_axis in grid])
		print(ngrid)	
		if ngrid <= Nsamp:
			# Sample at all the grid points if there are not too many
			random = False
			Ysamp = np.array([y for y in product(*grid)])

		else:	
			random = True
			Ysamp = np.zeros((Nsamp, m))
			
			# Generate random indices from the total number of points without replacement
			indices = []
			for k in range(Nsamp):
				# Find an index we have not yet used
				while True:
					newindex = randint(0, ngrid-1)
					if newindex not in indices:
						indices.append(newindex)
						break

				# Decode index into grid point
				ii = int(newindex)
				for j, grid_axis in enumerate(grid):
					# Product of the number of points per dimension below this
					nk = prod([ len(grid_axis_) for grid_axis_ in grid[j+1:]])
					# Get the remainder to extract the index in the current iterate
					jj = ii//nk
					Ysamp[k, j] = grid_axis[jj]
					# Extract the remainder
					ii -= jj*nk

		# Now determine what fraction of these epsilon balls are inside the domain 
		Ninside = 0
		z = cp.Variable(m)				# Point inside the domain
		y = cp.Parameter(m)				# grid point we are checking
		alpha = cp.Variable(len(Y))		# convex combination parameters
		# minimize distance between a point z that is a convex combination of the corners
		# and the grid center y
		obj = cp.Minimize(cp.norm(z - y))
		constraints = [z == alpha.__rmatmul__(Y.T), alpha >=0, cp.sum(alpha) == 1]
		prob = cp.Problem(obj, constraints)
		for yi in tqdm(Ysamp):
			y.value = yi
			res = prob.solve(warm_start = True)
			
			dist = np.linalg.norm(z.value - yi)
			#print(dist)
			if dist < eps:
				Ninside += 1

		if random:
			mean = float(Ninside/Nsamp)
			Ns[i] = int( mean*ngrid)
			Nstd[i] = np.sqrt(mean - mean**2)*float(ngrid) 
		else:
			Ns[i] = Ninside
			Nstd[i] = 0

		print('eps=%5.2g \t N=%12.7e' % (eps, Ns[i]))
		# Now save the data
		pgf = PGF()
		pgf.add('eps', epsilons[:i+1])
		pgf.add('N', Ns[:i+1])
		pgf.add('Nstd', Nstd[:i+1])
		pgf.write('data/fig_covering_%s.dat' % name)


		# Now compute rates
