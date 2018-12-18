import numpy as np
from psdr.demos import build_multif_domain

def test_multif():
	dom = build_multif_domain()
	# Check sampling
	dom.sample(10)

	# Check corners
	for i in range(10):
		p = np.random.randn(len(dom))
		x = dom.corner(p)
		#assert dom.extent(x, -p) > 0, "extent must be positive"
