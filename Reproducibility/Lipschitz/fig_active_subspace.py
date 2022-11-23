import numpy as np
import matplotlib.pyplot as plt
import psdr
import psdr.demos
from psdr.pgf import PGF

np.random.seed(0)

Us = []

fun = psdr.demos.OTLCircuit()

X = fun.domain.sample(2e2)
fX = fun(X) 
grads = fun.grad(X)

Xt = fun.domain.sample_grid(10)
fXt = fun(Xt) 

# Compare Lipschitz to the Active Subspace with gradients

fig, axes = plt.subplots(2,2, figsize = (8,8))

act = psdr.ActiveSubspace()
act.fit(grads)
act.shadow_plot(X, fX, dim = 1, ax = axes[0,0], pgfname = 'data/fig_active_subspace_as.dat')
act.shadow_envelope(Xt, fXt, ax = axes[0,0], pgfname = 'data/fig_active_subspace_as_shadow.dat')
axes[0,0].set_title('Active Subspace')

print act.U[:,0]
Us.append(np.copy(act.U[:,0]))

lip = psdr.LipschitzMatrix()
lip.fit(grads = grads)
lip.shadow_plot(X, fX, dim = 1, ax = axes[0,1], pgfname = 'data/fig_active_subspace_lipgrad.dat')
lip.shadow_envelope(Xt, fXt, ax = axes[0,1], pgfname = 'data/fig_active_subspace_lipgrad_shadow.dat')
axes[0,1].set_title("Lipschitz Matrix (gradient)")
print lip.U[:,0]
Us.append(np.copy(lip.U[:,0]))

# Now the same with sample information

# Ridge approximation
pra = psdr.PolynomialRidgeApproximation(degree = 5, subspace_dimension = 1)
pra.fit(X, fX)
pra.shadow_plot(X, fX, dim = 1, ax = axes[1,0], pgfname = 'data/fig_active_subspace_pra.dat')
pra.shadow_envelope(Xt, fXt, ax = axes[1,0], pgfname = 'data/fig_active_subspace_pra_shadow.dat')
axes[1,0].set_title("Polynomial Ridge Approximation")
print pra.U[:,0]
Us.append(np.copy(pra.U[:,0]))

lip.fit(X, fX)
lip.shadow_plot(X, fX, dim = 1, ax = axes[1,1], pgfname = 'data/fig_active_subspace_lipsamp.dat')
lip.shadow_envelope(Xt, fXt, ax = axes[1,1], pgfname = 'data/fig_active_subspace_lipsamp_shadow.dat')
axes[1,1].set_title("Lipschitz Matrix (samples)")
print lip.U[:,0]
Us.append(np.copy(lip.U[:,0]))

Us = np.array(Us)
print Us
pgf = PGF()
for i in range(6):
	pgf.add('x%d' % (i+1), Us[:,i])
pgf.write('data/fig_active_subspace_Us.dat')

fig.tight_layout()
plt.show()
