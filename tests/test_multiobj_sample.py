from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import psdr


def test_multiobj_sample():
	m = 2
	L1 = np.ones((1,m))
	L2 = np.ones((1,m))
	L2[0,1] = 0
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	Ls = [L1, L2]

	Xhat = []
	for i in range(10):
		x = psdr.seq_maximin_sample(dom, Xhat, Ls, Nsamp = 10)
		Xhat.append(x)
	
	Xhat = np.array(Xhat)

	# Now check fill distance
	for L in Ls:
		assert L.shape[0] == 1, "Only 1-d tests implemented"
		c1 =  dom.corner(L.flatten())
		c2 =  dom.corner(-L.flatten())
		lb, ub = sorted([L.dot(c1), L.dot(c2)])
		vol = float(ub - lb)
		d = psdr.fill_distance_estimate(dom, Xhat, L = L)
		print("")
		print("ideal fill distance ", 0.5*vol/(len(Xhat) - 1) ) 
		print("actual fill distance", d)

		# we add a fudge factor to ensure the suboptimal sampling passes 
		assert 0.25*d < 0.5*vol/(len(Xhat)-1), "Sampling not efficient enough"

#	print(np.sort(L1.dot(Xhat.T)).T)
#	print(np.sort(L2.dot(Xhat.T)).T)

if __name__ == '__main__':
	test_multiobj_sample()
