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
X = fun.domain.sample_grid(3)
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


# Scheme 3: optimal on ridge
u = Lmat.U[:,0]
c1 = fun.domain.corner(u)
c2 = fun.domain.corner(-u)
X3 = np.array([ t*c1 + (1-t)*c2 for t in np.linspace(0,1,64)])
R = psdr.sample_sphere(len(fun.domain), len(X3) - 2)
for i in range(1,len(X3)-1):
	dom_line = fun.domain.add_constraints(u, u.dot(X3[i]))
	print(R[i-1])
	X3[i] = dom_line.corner(R[i-1], verbose = True )

fX3 = fun(X)
name3 = 'line'


for X, fX, name in zip([X1, X3], [fX1, fX3], [name1, name3]):
	lb, ub = Lmat.bounds(X, fX, Xt)
	print("average uncertainty", np.mean(ub - lb)) 
	print("max uncertainty", np.max(ub - lb)) 
