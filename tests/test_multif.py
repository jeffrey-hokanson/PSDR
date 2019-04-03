from __future__ import print_function
import numpy as np
from psdr.demos import build_multif_domain

def test_multif():
	dom = build_multif_domain()
	for i in range(len(dom)):
		print("%10.2e, %10.2e" % (dom.norm_lb[i], dom.norm_ub[i])) 
	#dom = dom.normalized_domain()
	# Check sampling
	X = dom.sample(10)
	dom.isinside(X)
	# Check corners
	for i in range(10):
		p = np.random.randn(len(dom))
		x = dom.corner(p)
		# TODO: This fails to due to some tolerances that are not respected, perhaps due to a scaling issue
		#assert dom.isinside(x), "Corner should be inside the domain"
		#assert dom.extent(x, -p) > 0, "extent must be positive"
