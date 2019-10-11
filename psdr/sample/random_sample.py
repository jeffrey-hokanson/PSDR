from __future__ import division, print_function

import numpy as np

def random_sample(domain, N):
	r""" Randomly sample a specified domain according to its measure

	This is an alias to the :code:`domain.sample` method
	for consistency with other sampling methods.

	Parameters
	----------
	domain: Domain
		Domain on which to sample
	N: int
		Number of samples to take
	"""
	return domain.sample(N)
