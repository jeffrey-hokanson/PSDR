import matplotlib.pyplot as plt
import psdr
import psdr
import psdr.demos

fun = psdr.demos.OTLCircuit()

X = fun.domain.sample(2e2)
fX = fun(X) 
grads = fun.grad(X)

Xt = fun.domain.sample_grid(8)
fXt = fun(Xt) 

# Compare Lipschitz to the Active Subspace with gradients

fig, axes = plt.subplots(2,2, figsize = (8,8))

act = psdr.ActiveSubspace()
act.fit(grads)
act.shadow_plot(X, fX, dim = 1, ax = axes[0,0])
act.shadow_envelope(Xt, fXt, ax = axes[0,0])
axes[0,0].set_title('Active Subspace')

print act.U[:,0]

plt.show()

lip = psdr.LipschitzMatrix()
lip.fit(grads = grads)
lip.shadow_plot(X, fX, dim = 1, ax = axes[0,1])
lip.shadow_envelope(Xt, fXt, ax = axes[0,1])
axes[0,1].set_title("Lipschitz Matrix (gradient)")
print lip.U[:,0]

# Now the same with sample information

# Ridge approximation
pra = psdr.PolynomialRidgeApproximation(degree = 5, subspace_dimension = 1)
pra.fit(X, fX)
pra.shadow_plot(X, fX, dim = 1, ax = axes[1,0])
pra.shadow_envelope(Xt, fXt, ax = axes[1,0])
axes[1,0].set_title("Polynomial Ridge Approximation")
print pra.U[:,0]

lip.fit(X, fX)
lip.shadow_plot(X, fX, dim = 1, ax = axes[1,1])
lip.shadow_envelope(Xt, fXt, ax = axes[1,1])
axes[1,1].set_title("Lipschitz Matrix (samples)")
print lip.U[:,0]

fig.tight_layout()
plt.show()
