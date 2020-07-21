import numpy as np
import psdr


def test_voronoi_vertex(m = 2, M = 10):
	domain = psdr.BoxDomain(-1*np.ones(m), np.ones(m))
	Xhat = domain.sample_grid(2)
	
	V = psdr.voronoi_vertex(domain, Xhat)

if __name__ == '__main__':
	test_voronoi_vertex()
