import numpy as np
from psdr import BaseFunction, minimax


def test_minimax():
	fun = BaseFunction()
	fun.eval = lambda x: np.array([x[0] + x[1], x[0]**2 + x[1]**2])
	fun.grad = lambda x: np.array([ [1, 1], [2*x[0], 2*x[1]]])
	x0 = 0.5*np.ones(2)
	x = minimax(fun, x0)
	assert np.all(np.abs(x - np.zeros(2) < 2e-5))
