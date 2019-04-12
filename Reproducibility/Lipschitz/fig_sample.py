from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

import psdr
import psdr.demos
from psdr.pgf import PGF

np.random.seed(0)
fig, axes = plt.subplots(3,1, figsize = (10,5))

fun = psdr.demos.OTLCircuit()


# Compute Lipschitz constant and Lipschitz matrix
X = fun.domain.sample_grid(2)
gradX = fun.grad(X)

Lmat = psdr.LipschitzMatrix()
Lcon = psdr.LipschitzConstant()

Lmat.fit(grads = gradX)
Lcon.fit(grads = gradX)


# Function evaluations for emprical shadow plot
Xt = fun.domain.sample_grid(5)
fXt = fun(Xt) 



# Scheme 1: Corners
X1 = fun.domain.sample_grid(2)
fX1 = fun(X)
name1 = 'corner'

# Scheme 2: maximin with Lipschitz matrix
X2 = [fun.domain.corner(Lmat.U[:,0]), fun.domain.corner(-Lmat.U[:,0])]
print("Lipschitz matrix sample")
while len(X2) < len(X1):
	X2.append(psdr.seq_maximin_sample(fun.domain, X2, L = Lmat.L, Nsamp = 1000))
	print(' '.join(['%8.3f' % x for x in X2[-1]]))
fX2 = fun(X2)
name2 = 'Lmat'

# Scheme 3: maximin with Lipschitz constant
X3 = [fun.domain.corner(Lmat.U[:,0]), fun.domain.corner(-Lmat.U[:,0])]
print("Lipschitz const sample")
while len(X3) < len(X1):
	X3.append(psdr.seq_maximin_sample(fun.domain, X3, L = Lcon.L, Nsamp = 1000))
	print(' '.join(['%8.3f' % x for x in X3[-1]]))
fX3 = fun(X3)
name3 = 'Lcon'

# Scheme 4: optimal on ridge
u = Lmat.U[:,0]
c1 = fun.domain.corner(u)
c2 = fun.domain.corner(-u)
X4 = np.array([ t*c1 + (1-t)*c2 for t in np.linspace(0,1,64)])

fX4 = fun(X4)
name4 = 'line'


for X, fX, name in zip([X1, X2, X3, X4], [fX1, fX2, fX3, fX4], [name1, name2, name3, name4]):
	print(" === %10s === " % name)
	lb, ub = Lmat.bounds(X, fX, Xt)
	print("average uncertainty", np.mean(ub - lb)) 
	print("max uncertainty", np.max(ub - lb))
	
	if True:
		Lmat.shadow_envelope_estimate(fun.domain, X, fX, 
			pgfname = 'data/fig_sample_Lmat_%s_envelope_estimate.dat' % name, progress = 2, ngrid = 100)

		Lmat.shadow_envelope_estimate(fun.domain, X, fX, 
			pgfname = 'data/fig_sample_Lcon_%s_envelope_estimate.dat' % name, progress = 2, ngrid = 100)
		  
		Lmat.shadow_envelope(Xt, fXt, pgfname = 'data/fig_sample_%s_envelope.dat' % name, ngrid = 100)

