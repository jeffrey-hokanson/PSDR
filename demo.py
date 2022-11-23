import numpy as np
import psdr
import psdr.demos
from psdr.initialization import initialize_subspace
from psdr.local_linear import local_linear_grads
from psdr.demos.polynomial import QuadraticFunction

#fun = psdr.demos.Borehole()
#X = fun.domain.sample(100)
#fX = fun(X)
#grads = fun.grad(X[0:10])

#U = initialize_subspace(X = X, fX = fX, grads = grads)
#print(U)
#print(U.shape)

fun = QuadraticFunction()
X = fun.domain.sample(20)
fX = fun(X)
print(fX)
print(fX.shape)
grads = fun.grad(X)
print(grads)
print(grads.shape)

