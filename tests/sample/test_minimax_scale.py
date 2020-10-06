import numpy as np
import psdr


def test_minimax_scale():
	M = 15
	m = 2

	domain = psdr.BoxDomain(-np.ones(m), np.ones(m))	

	Xhat = psdr.maximin_block(domain, M)

	psdr.minimax_scale(domain, Xhat)


if __name__ == '__main__':
	test_minimax_scale()
