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
		'polyrat',
	]

install_requires += [
	'matplotlib',
	'scipy>=1.1.0',
	]

test_requires = ['pytest']


with open("README.md", "r") as fh:
    long_description = fh.read()

ns = {}
with open('psdr/version.py') as f:
	exec(f.read(), ns)

version = ns['__version__']

setup(name='psdr',
	version = version,
	description = 'Parameter Space Dimension Reduction Toolbox',
    long_description=long_description,
    long_description_content_type="text/markdown",
	author = 'Jeffrey M. Hokanson',
	author_email = 'jeffrey@hokanson.us',
	url = 'https://github.com/jeffrey-hokanson/PSDR',
	packages = ['psdr', 'psdr.demos', 'psdr.domains', 'psdr.sample', 'psdr.geometry'],
	install_requires = install_requires,
	test_requires = test_requires,
	classifiers = [
		'Development Status :: 4 - Beta',
		'Intended Audience :: Science/Research',
		'License :: OSI Approved :: GNU Affero General Public License v3',
		'Operating System :: OS Independent',
		'Programming Language :: Python :: 3.6',
		'Programming Language :: Python :: 3.7',
		'Programming Language :: Python :: 3.8',
		'Topic :: Scientific/Engineering :: Mathematics'
		]
	)
