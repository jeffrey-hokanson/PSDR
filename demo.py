import numpy as np
import psdr
import psdr.demos

fun = psdr.demos.OTLCircuit()

X = fun.domain.sample_grid(2)
X = np.vstack([X, fun.domain.sample(5)])
L1 = np.ones((1, len(fun.domain)))
L2 = np.zeros((1,len(fun.domain)))
L2[0,3] = 1.
Ls = [L1, L2]

x = psdr.seq_maximin_sample(fun.domain, X, Ls = Ls)

