""" Utilities for running trivially parallel code


Note there are several options depending on level of complexity:
+ Pathos, pathos.multiprocessing



"""
from __future__ import print_function

import numpy as np
try:
	from tqdm import tqdm
	PROGRESS = True
except:
	PROGRESS = False

import warnings
from copy import deepcopy

try:
	# NB: This is the multiprocessing library from Pathos
	from multiprocess import Pool
	MULTIPROCESS = True
except:
	warnings.warn("Install pathos multiprocess library to enable parallel map")
	MULTIPROCESS = False

def pmap(f, args = None, kwargs = None, processes = None, progress = False, parallel = True, desc = None):
	""" Parallel map

	Efficiently computes

		f(*args[i], **kwargs[i]) for i in range(min(len(args), len(kwargs)))

	WARNING: This has been seen to cause crashes when called within a class

	
	Parameters
	----------
	f: callable
		f will be called with in the form
	"""
	progress = progress & PROGRESS

	inputs = []
	if kwargs is None:
		inputs = [ (args_, {}) for args_ in args]
	elif isinstance(kwargs, dict):
		inputs = [ (args_, deepcopy(kwargs)) for args_ in args]
	elif isinstance(kwargs[0], dict):
		inputs = [ (args_, kwargs_) for args_, kwargs_ in zip(args, kwargs)]
	else:
		raise NotImplementedError

	if MULTIPROCESS and parallel:
		pool = Pool(processes = processes)
		if progress:
			pbar = tqdm(total = len(inputs), desc = desc)

			def callback(args):
				pbar.update(1)
		else:
			callback = None
		
		results = [pool.apply_async(f, args, kwargs, callback = callback ) for  (args, kwargs) in inputs] 
		results = [result.get() for result in results]
		pool.close()
		pool.join()

		if progress:
			pbar.close()
	else:
		results = []
		if progress:
			for args, kwargs in tqdm(inputs, total = len(inputs), desc = desc):
				results.append(f(*args, **kwargs))
		else:	
			for args, kwargs in inputs:
				results.append(f(*args, **kwargs))
		
	return results


if __name__ == '__main__':
	import time

	class Dummy():
		def __init__(self, x):
			self.x = x

	def f(dummy):
		time.sleep(0.2)
		return np.linalg.norm(dummy.x)

	args = [ (Dummy(np.random.randn(10000, 100)),) for i in range(100)]
	
	print(pmap(f, args, progress = True, parallel = True))


