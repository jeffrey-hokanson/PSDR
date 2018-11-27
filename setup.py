import os
from setuptools import setup


setup(name='psdr',
	version = '0.1',
	description = 'Parameter Space Dimension Reduction Toolbox',
	author = 'Jeffrey M. Hokanson',
	packages = ['psdr', 'psdr.opt', 'psdr.demos', 'psdr.util'],
	install_requires = [
		'numpy', 
		'scipy', 
		'matplotlib<=2.2.3',
		],
	)
