import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import psdr
from psdr.pgf import PGF

np.random.seed(4)
# Places to sample
X = np.random.uniform(-1,1, size = (10,1))
X = X[np.argsort(X.flatten())]
X[5] = (1-0.1)/3.
X[6] = (1+0.3)/3.
X[8] = 0.75
X[9] = -0.65
print X

f = lambda x: np.sin(3*np.pi*x).flatten()
fX = f(X)

# Places to sample
Xtest = np.linspace(-1,1,500).reshape(-1,1)
xx = Xtest.flatten()

# Radial basis function approximation
kernel = RBF(1, (1e-2, 1e2))
gpr = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)
gpr.fit(X, fX)
print gpr.kernel_.get_params()

yy_gpr, yy_std = gpr.predict(Xtest, return_std = True)

# Lipschitz fit
lip = psdr.LipschitzMatrix()
lip.fit(X, fX)
print(lip.L)
lb, ub = lip.bounds(X, fX, Xtest)
yy_lip =  np.mean([lb,ub], axis = 0)


# Plotting
fig, axes = plt.subplots(1,2, figsize = (10,4))

ax = axes[0]
ax.set_title('Gaussian Process')
ax.plot(X.flatten(), fX, 'k.')
ax.plot(xx, f(xx), 'b-')
ax.plot(xx, yy_gpr, 'r-')
ax.fill_between(xx.flatten(), yy_gpr - yy_std, yy_gpr + yy_std, color = 'g', alpha = 0.5)


ax = axes[1]
ax.set_title('Lipschitz Bounds')

ax.plot(X.flatten(), fX, 'k.')
ax.plot(xx, f(xx), 'b-')
ax.plot(xx, yy_lip, 'r-')
ax.fill_between(xx.flatten(), lb, ub, color = 'g', alpha = 0.5)


axes[0].set_ylim(axes[1].get_ylim())

fig.tight_layout()
plt.show()

# Save PGF data

pgf = PGF()
pgf.add('x', X.flatten())
pgf.add('fx', fX)
pgf.write('data/fig_gp_data.dat')

pgf = PGF()
pgf.add('x', xx.flatten())
pgf.add('fx', f(xx))
pgf.add('gpr', yy_gpr)
pgf.add('gpr_lb', yy_gpr - yy_std)
pgf.add('gpr_ub', yy_gpr + yy_std)
pgf.add('lip_lb', lb)
pgf.add('lip_ub', ub)
pgf.write('data/fig_gp.dat')
