import os
import sys
from setuptools import setup

install_requires = [
		'numpy>=1.15', 
		'scipy>=1.0.0', 
		'redis',
		'cvxpy',
		'cvxopt',
		'tqdm',
		'dask',
		'distributed',
		'sobol_seq',
		'satyrn>=0.3.2',
		'osqp',
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
	version = '0.2',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.demos', 'psdr.domains', 'psdr.sample', 'psdr.geometry'],
	install_requires = install_requires,
	)
