from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['CoordinateBasedDimensionReduction']

class CoordinateBasedDimensionReduction(object):
	r""" Parent class for dimension reduction strategies that select variables
	"""

	@property
	def score(self):
		r""" The score associated with each parameter
		"""
		return self._score

	def plot_score(self, ax = 'auto', domain = None):
		if ax is 'auto':
			fig, ax = plt.subplots(figsize = (6,6))	 
		
		if ax is not None:
			x = np.arange(0, len(self.score))
			ax.bar(x, self.score, align = 'center')
			ax.set_xticks(x)
			if domain is not None:
				ax.set_xticklabels(domain.names, rotation = 90)
			ax.set_xlabel('parameter')
			ax.set_ylabel('score')
