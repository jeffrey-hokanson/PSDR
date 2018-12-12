# Subspace based dimension reduction techniques
import numpy as np
import matplotlib.pyplot as plt

class SubspaceBasedDimensionReduction(object):
	@property
	def U(self):
		""" A matrix defining the 'important' directions

		Returns an np.array of dimension (m,m)
		"""
		raise NotImplementedError


	def shadow_plot(self, X, fX, dim = 1, ax = None):
		r""" Draw a shadow plot


		"""
		if ax is None:
			fig, ax = plt.subplots()
		print X.shape
		print self.U.shape
		ax.plot(X.dot(self.U[:,0]), fX)


class ActiveSubspace(SubspaceBasedDimensionReduction):
	r"""Computes the active subspace based on gradient samples

	

	"""
	def __init__(self, grads, weights = None):
		self._grads = np.array(grads)
		self._U, s, VT = np.linalg.svd(self._grads.T)

	@property
	def U(self):
		return self._U


if __name__ == '__main__':
	X = np.random.randn(100,5)
	a = np.random.randn(5,)
	fX = np.dot(X, a).flatten()
	grads = np.tile(a, (X.shape[0], 1))
	act = ActiveSubspace(grads)
	act.shadow_plot(X, fX)
	plt.show()
