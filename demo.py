import numpy as np
import psdr
import psdr.demos
from psdr.opg import *

fun = psdr.demos.OTLCircuit()
X = fun.domain.sample(500)
fX = fun(X).flatten()
opg = OuterProductGradient()
opg.fit(X, fX)


grads = fun.grad(X)
act = psdr.ActiveSubspace()
act.fit(grads)

print('opg', opg.U[:,0])
print('act', act.U[:,0])
