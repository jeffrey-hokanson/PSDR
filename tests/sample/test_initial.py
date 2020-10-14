import numpy as np
import psdr


def test_initial():
	np.random.seed(0)
	m = 5
	L = np.random.randn(1,m)
	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))

	X0 = psdr.initial_sample(domain, L, 10)
	print(X0)	


if __name__ == '__main__':
	test_initial()
