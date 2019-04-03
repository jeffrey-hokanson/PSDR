import os
import sys
from setuptools import setup


if sys.version_info[0] < 3:
	install_requires = [
		'numpy>=1.15', 
		'scipy', 
		'matplotlib<3.0.0',
		'redis',
		'dill',
		'cvxpy',
		'cvxopt',
		'tqdm',
		'dask',
		'distributed',
		# Required for python 2.7
		'tornado<6.0.0',
		]
else:
	install_requires = [
		'numpy>=1.15', 
		'scipy', 
		'matplotlib',
		'redis',
		'dill',
		'cvxpy',
		'cvxopt',
		'tqdm',
		'dask',
		'distributed',
		]


setup(name='psdr',
	version = '0.1',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.demos'],
	install_requires = install_requires,
	)
