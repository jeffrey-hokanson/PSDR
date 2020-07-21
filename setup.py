import os
import sys
from setuptools import setup

install_requires = [
		'numpy>=1.15', 
		'redis',
		'cvxpy',
		'cvxopt',
		'tqdm',
		'dask',
		'distributed',
		'sobol_seq',
		'satyrn>=0.3.2',
		'iterprinter',
	]

install_requires += [
	'matplotlib',
	'scipy>=1.1.0',
	]



setup(name='psdr',
	version = '0.3.6',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.demos', 'psdr.domains', 'psdr.sample', 'psdr.geometry'],
	install_requires = install_requires,
	)
