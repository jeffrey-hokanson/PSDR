from __future__ import print_function, division

import numpy as np
import sobol_seq
	
from ..domains import RandomDomain	
	
def sobol_sequence(domain, n):
	r""" Generate samples from a Sobol sequence on a domain

	A Sobol sequence is a low-discrepancy sequence [Sobol_wiki]_;
	here we generate this sequence using a library descended from 
	ACM TOMS algorithms 647 and 659 [sobol_seq]_. 
	Although Sobol sequences generated on :math:`[0,1]^m`, for domains 
	that are not a simple box, here we simply reject those samples
	falling outside the domain (after scaling). 

	*Warning*: when the domain occupies only a small fraction of its enclosing
	hypercube, this function can take a while to execute.


	Parameters
	----------
	domain: Domain
		Domain on which to construct the Sobol sequence
	
	n: int
		Number of elements from the Sobol sequence to return

	Returns
	-------
	np.array (n, len(domain))
		The samples from the Sobol sequence inside the domain	

	References
	----------
	.. [Sobol_wiki] (Sobol Sequence)[https://en.wikipedia.org/wiki/Sobol_sequence]	
	.. [sobol_seq] https://github.com/naught101/sobol_seq
	"""
	assert len(domain.A_eq) == 0, "Currently equality constraints on the domain are not supported"
	assert not isinstance(domain, RandomDomain), "Currently does not support random domains"

	skip = 1
	Xsamp = np.zeros((0, len(domain)))
	while True:
		# Generate a Sobol sequence on the cube [0,1]^m
		X_norm = sobol_seq.i4_sobol_generate(len(domain), n, skip = skip) 
		skip += n
		
		# Scale into the domain
		X = (domain.norm_ub - domain.norm_lb)*X_norm + domain.norm_lb	
		
		# Add only those points inside the domain
		Xsamp = np.vstack([Xsamp, X[domain.isinside(X)]])
		if Xsamp.shape[0] >= n:	
			break

	return Xsamp[0:n]

