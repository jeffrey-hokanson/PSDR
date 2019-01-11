import numpy as np
from .domains import EmptyDomain
from sklearn.neighbors import KernelDensity
from .pgf import PGF


def forward_propagation_pdf(random_domain, pra_rand_norm, Nsamp = int(1e4), npoints = 100):
	Xrand = random_domain.sample(Nsamp)
	Xrand_norm = random_domain.normalize(Xrand)
	y_rand = pra_rand_norm.predict(Xrand_norm)
	
	p0, p25, p75, p100 = np.percentile(y_rand, [0,25,75,100])
	rng = p75 - p25
	kde = KernelDensity(kernel = 'linear', bandwidth = rng/10.)
	kde.fit(y_rand.reshape(-1,1))
	
	xx = np.linspace(p0, p100, npoints)
	yy = np.exp(kde.score_samples(xx.reshape(-1,1)))
	return xx, yy

def plot_pgf_forward_propagation_pdf(fname, random_domain, pra_rand_norm, **kwargs):	
	xx, yy = forward_propagation_pdf(random_domain, pra_rand_norm, **kwargs)
	pgf = PGF()
	pgf.add('x', xx)
	pgf.add('y', yy)
	pgf.write(fname)

class RidgeChanceConstraint(object):
	"""

		if mode == 'upper':
			P_{x_r} [ pra_norm(x_norm_d, x_norm_r) < threshold] < prob
		elif mode == 'lower':
			P_{x_r} [ pra_norm(x_norm_d, x_norm_r) > threshold] < prob

	"""
	def __init__(self, design_domain, random_domain, pra_norm, threshold, mode = 'upper', prob = 1e-2):
		self.design_domain = design_domain
		self.random_domain = random_domain
		self.design_domain_norm = design_domain.normalized_domain()
		self.random_domain_norm = random_domain.normalized_domain()
		self.pra_norm = pra_norm
		self.threshold = threshold
		self.mode = mode
		self.prob = prob

	def deterministic_failure_points_1d(self):
		""" Compute the input where ridge function crosses the failure boundary
		""" 
		assert self.pra_norm.U.shape[1] == 1, "Only works for 1-d problems"
		roots = self.pra_norm.roots(val = self.threshold)
		# return only those roots inside the domain
		interior_roots = []
		for r in roots:
			try:
				dom = self.design_domain_norm.add_constraint(A_eq = self.pra_norm.U[:len(self.design_domain)].T, b_eq = np.array([r]))
				interior_roots.append(r)
			except EmptyDomain:
				pass
		return interior_roots
	

	def sample_random_ridge(self, Nsamp = int(1e4)):
		U = self.pra_norm.U[-len(self.random_domain):]

		Xrand = self.random_domain.sample(1e4)
		Xrand_norm = self.random_domain.normalize(Xrand)

		yrand = np.dot(U.T, Xrand_norm.T).T
		return yrand	

	def empirical_safety_factor_1d(self):
		assert self.pra_norm.U.shape[1] == 1, "Only works for 1-d problems"
		y_rand = self.sample_random_ridge()

		if self.mode == 'upper':
			return float(np.percentile(y_rand.flatten(), 100*(1. - self.prob)))
		else:
			return float(np.percentile(y_rand.flatten(), 100.*self.prob))
		
	def random_ridge_pdf(self, npoints = 100, **kwargs):
		assert self.pra_norm.U.shape[1] == 1, "Only works for 1-d problems"
		y_rand = self.sample_random_ridge(**kwargs)
		
		p0, p25, p75, p100 = np.percentile(y_rand, [0,25,75,100])
		rng = p75 - p25
		kde = KernelDensity(kernel = 'linear', bandwidth = rng/10.)
		kde.fit(y_rand.reshape(-1,1))
		
		xx = np.linspace(p0, p100, npoints)
		yy = np.exp(kde.score_samples(xx.reshape(-1,1)))
		return xx, yy

	def plot_pgf_random_ridge_pdf(self, fname, **kwargs):
		xx , yy = self.random_ridge_pdf(**kwargs)
		pgf = PGF()
		pgf.add('x', xx)
		pgf.add('y', yy)
		pgf.write(fname)

