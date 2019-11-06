import numpy as np
import psdr

def test_projection_design():
	m = 5
	M = 10
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	L1 = np.ones((1,m))

	if False:
		L2 = np.ones((2,m))
		L2[0,0] = -1.
		L2[1,1] = -1.
	else:
		L2 = np.ones((1,m))
		L2[0,0] = -1.

	X = psdr.projection_design(dom, M, [L1, L2])
	print(X)	

if __name__ == '__main__':
	test_projection_design()
