
class BoxDomain(LinIneqDomain):
	r""" Implements a domain specified by box constraints

	Given a set of lower and upper bounds, this class defines the domain

	.. math::

		\mathcal{D} := \lbrace \mathbf{x} \in \mathbb{R}^m : \text{lb} \le \mathbf{x} \le \text{ub} \rbrace \subset \mathbb{R}^m.

	Parameters
	----------
	lb: array-like (m,)
		Lower bounds
	ub: array-like (m,)
		Upper bounds
	"""
	def __init__(self, lb, ub, names = None):
		LinQuadDomain.__init__(self, lb = lb, ub = ub, names = names)	
		assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), "Both lb and ub must be finite to construct a box domain"

	# Due to the simplicity of this domain, we can use a more efficient sampling routine
	def _sample(self, draw = 1):
		x_sample = np.random.uniform(self.lb, self.ub, size = (draw, len(self)))
		return x_sample

	def _corner(self, p):
		# Since the domain is a box, we can find the corners simply by looking at the sign of p
		x = np.copy(self.lb)
		I = (p>=0)
		x[I] = self.ub[I]
		return x

	def _extent(self, x, p):
		return self._extent_bounds(x, p)

	def _isinside(self, X, tol = TOL):
		return self._isinside_bounds(X, tol = tol)

	def _normalized_domain(self, **kwargs):
		names_norm = [name + ' (normalized)' for name in self.names]
		return BoxDomain(lb = self.lb_norm, ub = self.ub_norm, names = names_norm)

	@property
	def A(self): return np.zeros((0,len(self)))
	
	@property
	def b(self): return np.zeros((0))
	
	@property
	def A_eq(self): return np.zeros((0,len(self)))
	
	@property
	def b_eq(self): return np.zeros((0))
	
	
	def latin_hypercube(self, N, metric = 'maximin', maxiter = 100, jiggle = False):
		r""" Generate a Latin-Hypercube design

		

		This implementation is based on [PyDOE](https://github.com/tisimst/pyDOE/blob/master/pyDOE/doe_lhs.py). 


		Parameters
		----------
		N: int
			Number of samples to take
		metric: ['maximin', 'corr']
			Metric to use when selecting among multiple Latin Hypercube designs. 
			One of
		
			- 'maximin': Maximize the minimum distance between points, or 
			- 'corr' : Minimize the correlation between points.

		jiggle: bool, default False
			If True, randomize the points within the grid specified by the 
			Latin hypercube design.

		maxiter: int, default: 100
			Number of random designs to generate in attempting to find the optimal design 
			with respect to the metric.
		"""	

		N = int(N)
		assert N > 0, "Number of samples must be positive"
		assert metric in ['maximin', 'corr'], "Invalid metric specified"

		xs = []
		for i in range(len(self)):
			xi = np.linspace(self.norm_lb[i], self.norm_ub[i], N + 1)
			xi = (xi[1:]+xi[0:-1])/2.
			xs.append(xi)

		# Higher score == better
		score = -np.inf
		X = None
		for it in range(maxiter):
			# Select which components of the hypercube
			I = [np.random.permutation(N) for i in range(len(self))]
			# Generate actual points
			X_new = np.array([ [xs[i][j] for j in I[i]] for i in range(len(self))]).T
	
			# Here we would jiggle points if so desired
			if jiggle:
				for i in range(len(self)):
					h = xs[i][1] - xs[i][0] # Grid spacing
					X_new[:,i] += np.random.uniform(-h/2, h/2, size = N)	

			# Evaluate the metric
			if metric == 'maximin':
				new_score = np.min(pdist(X_new))
			elif metric == 'corr':
				new_score = -np.linalg.norm(np.eye(len(self)) - np.corrcoef(X_new.T))

			if new_score > score:
				score = new_score
				X = X_new

		return X
