=============================
Multiprocessing
=============================

Successfully implementing multiprocessing with user defined functions 
has a bit of a rough edge in Python. There are three main use cases 
we see for our users

	1. Simple models that are algebraic relationships 
	2. Complex python-based models requiring large libraries
	3. Complex compiled models based on external libraries

The first and last of these, although very different, look similar 
on the Python side: they are simple functions with no dependencies 
