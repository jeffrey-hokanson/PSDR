import numpy as np
import psdr
import psdr.demos
from psdr.subspace_init import *
from psdr.local_linear import local_linear_grad

fun = psdr.demos.Borehole()
X = fun.domain.sample(100)
fX = fun(X)

grads = local_linear_grad(X, fX)
print(grads)
#U = subspace_init(2, X = X, fX = fX)
#print(U)
