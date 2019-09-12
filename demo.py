import numpy as np
import psdr
import psdr.demos
from psdr.subspace_init import *

fun = psdr.demos.Borehole()
X = fun.domain.sample(2000)
fX = fun(X)

U = subspace_init(2, X = X, fX = fX)
print(U)
