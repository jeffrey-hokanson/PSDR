import numpy as np
import psdr
from psdr.opg import *


X = np.random.randn(100,3)
fX = np.random.randn(len(X))

opg = OuterProductGradient()
opg.fit(X, fX)


