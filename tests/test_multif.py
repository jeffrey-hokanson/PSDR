from __future__ import print_function
import numpy as np
from psdr.demos import build_multif_domain, MULTIF


def test_multif_domain():
	np.random.seed(0)
	print("Building domain")
	dom = build_multif_domain().normalized_domain()

	# Check sampling
	print("sampling")
	X = dom.sample(10)
	dom.isinside(X)

	# Check corners
	for i in range(10):
		print("iter %d" % i)
		p = np.random.randn(len(dom))
		x = dom.corner(p)
		#for xi, lbi, ubi, name in zip(x, dom.lb, dom.ub, dom.names):
		#	print("%10.7f %10.7f %10.7f" % (xi, lbi, ubi), name)
		# TODO: This fails to due to some tolerances that are not respected, perhaps due to a scaling issue
		assert dom.isinside(x), "Corner should be inside the domain"
		assert dom.extent(x, -p) > 0, "extent must be positive"

def test_multif():
	multif = MULTIF()
	x = multif.domain.sample()
	y = multif(x, verbose = True)
	print(y)

if __name__ == '__main__':
	test_multif()
	#test_multif_domain()
