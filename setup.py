import os
from setuptools import setup


setup(name='psdr',
	version = '0.1',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.opt', 'psdr.demos', 'psdr.util'],
	install_requires = [
		'numpy>=1.14', 
		'scipy<=1.1.0', 
		'matplotlib<=2.2.3',
		'cvxopt',
		'redis',
		'dill',
		'cvxpy',
		],
	)
