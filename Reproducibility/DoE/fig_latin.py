from __future__ import print_function

import numpy as np
from scipy.linalg import orth
import matplotlib.pyplot as plt
import psdr 

np.random.seed(0)

dom = psdr.BoxDomain(-np.ones(2), np.ones(2))

fig, axes = plt.subplots(1, 2, figsize = (10, 5))

# Number of samples
M = 10

# Latin Hypercube sampling
X = dom.latin_hypercube(M)

ax = axes[0]
ax.plot(X[:,0], X[:,1], 'k.')
ax.set_title('Latin Hypercube')


# Lipschitz-based sampling
#Ls = [orth(np.ones((2,1))).T, np.array([[1],[0]]).T]
#Ls = [np.array([[0],[1]]).T, np.array([[1],[0]]).T]
Ls = [orth(np.ones((2,1))).T]
print(Ls)
X = []
for i in range(10):
	x = psdr.multiobj_seq_maximin_sample(dom, X, Ls)
	X.append(x)
X = np.vstack(X)

ax = axes[1]
ax.plot(X[:,0], X[:,1], 'k.')
ax.set_title('Lipschitz Matrix')

for ax in axes:
	ax.set_xlim(-1.05,1.05)
	ax.set_ylim(-1.05,1.05)
plt.show()
