Function Class
==============

The function class provides a wrapper around 
a function :math:`f`
posed on some domain :math:`\mathcal{D} \subset \mathbb{R}^m`.


Design Considerations
---------------------

Often-times functions coming from engineering applications 
come in *application units*, e.g., in meters, Pascals, etc.
The trouble is that these units are often poorly scaled with respect
to each other, causing numerical issues.
Hence, one of the first steps is to restate the problem on the *normalized domain*:
an affine transform of :math:`\mathcal{D}` into the unit box :math:`[-1,1]^m`. 
One of the roles of the :code:`Function` class is to transparently handle
working in the normalized domain so that no changes are needed for the functions provided. 



Function Class API
------------------

.. autoclass:: psdr.Function


