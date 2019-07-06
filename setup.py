import os
import sys
from setuptools import setup

install_requires = [
		'numpy>=1.15', 
		'scipy', 
		'redis',
		'cvxpy',
		'cvxopt',
		'tqdm',
		'dask',
		'distributed',
		'sobol_seq',
		'pycosat',
	]

if sys.version_info[0] < 3:
	install_requires += [
		'matplotlib<3.0.0',
		# Required for python 2.7 from dask distributed
		'tornado<6.0.0',
		# LRU Cache
		'backports.functools_lru_cache',
		]
else:
	install_requires += [
		'matplotlib',
		]



setup(name='psdr',
	version = '0.1',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.demos'],
	install_requires = install_requires,
	)
