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
	version = '0.3.7',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	author_email = 'jeffrey@hokanson.us',
	url = 'https://github.com/jeffrey-hokanson/PSDR',
	packages = ['psdr', 'psdr.demos', 'psdr.domains', 'psdr.sample', 'psdr.geometry'],
	install_requires = install_requires,
	)
