import os
from setuptools import setup


setup(name='psdr',
	version = '0.1',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.opt', 'psdr.demos'],
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
		],
	)
