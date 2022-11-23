=======================
Demonstration Functions
=======================

To aid in exploration we include a variety of test functions
that can be used in conduction with the 
parameter space dimension reduction techniques.
We divide these into two classes:
those that are simple formulas
and those that are the results of the numerical approximation 
of differential equations.
The former are inexpensive to evaluate and equipped with analytic gradients;
the latter can be expensive, may not have gradients, and may contain computational noise.

We provide two different ways to access these demo functions:
a :meth:`psdr.Function` interface working on the normalized domain
and a low-level access to the underlying function and domain methods.

Formula-based Test Functions
============================


Borehole Function
-----------------

.. autoclass:: psdr.demos.Borehole

..
	.. autofunction:: psdr.demos.borehole
	.. autofunction:: psdr.demos.borehole_grad
	.. autofunction:: psdr.demos.build_borehole_domain
	.. autofunction:: psdr.demos.build_borehole_uncertain_domain

Golinski Gearbox
----------------

.. autoclass:: psdr.demos.GolinskiGearbox

Nowacki Beam
------------

.. autoclass:: psdr.demos.NowackiBeam

OTL Circuit Function
--------------------

.. autoclass:: psdr.demos.OTLCircuit

Piston Function 
---------------

.. autoclass:: psdr.demos.Piston

Robot Arm Function
------------------

.. autoclass:: psdr.demos.RobotArm

Wing Weight Function
--------------------

.. autoclass:: psdr.demos.WingWeight



Model-based Test Functions
==========================


OpenAeroStruct
--------------

.. autoclass:: psdr.demos.OpenAeroStruct


MULTI-F
-------

.. autoclass:: psdr.demos.MULTIF


..  
	.. autofunction:: psdr.demos.multif
	.. autofunction:: psdr.demos.build_multif_design_domain
	.. autofunction:: psdr.demos.build_multif_random_domain

NACA0012
--------

.. autoclass:: psdr.demos.NACA0012


