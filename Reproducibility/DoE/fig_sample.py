import numpy as np
import psdr, psdr.demos
from psdr.pgf import PGF

fun = psdr.demos.OTLCircuit()

# Construct estimate of Lipschitz matrix
Xg = fun.domain.sample_grid(2)
grads = fun.grad(Xg)
lip = psdr.LipschitzMatrix()
lip.fit(grads = grads)
L = lip.L


# Construct the testing dataset
Xt = fun.domain.sample_grid(5)
ft = fun(Xt).flatten()
scale = np.max(ft) - np.min(ft)
samplers = {
	# should all take two arguments: the domain and the number of samples
	'random': psdr.random_sample,
	'maximin': lambda dom, N : psdr.maximin_sample(dom, N),
	'maximin_lip': lambda dom, N : psdr.maximin_sample(dom, N, L = lip.L),
}

responses = {
	# Response surfaces to try out
	'pra': psdr.PolynomialRidgeApproximation(subspace_dimension = 1, degree = 5),
	'gp_iso': psdr.GaussianProcess(degree = 0),
}

np.random.seed(0)

Nvec = [10,20,30,40,50,60,70,80,90,100]
Niter = 20
for samp_name in samplers:
	samp = samplers[samp_name]
	print("Sampler: %s" % samp_name)	
	# Place to store the data
	data = {key: np.zeros((len(Nvec), Niter)) for key in responses}

	for i, N in enumerate(Nvec):
		for it in range(Niter):
			# Construct sampling scheme
			print("Building %4d point %15s design, iteration %3d" % (N, samp_name, it))
			X = samp(fun.domain, N)
			fX = fun(X)
			for resp_name in responses:	
				resp = responses[resp_name]
				# Now fit and test response surface
				resp.fit(X, fX)
				# Record the sup norm error
				data[resp_name][i, it] = np.max(np.abs(resp(Xt).flatten() - ft.flatten() ))/scale
				print("\t err %10s: %8.2e" % (resp_name, data[resp_name][i,it]))

		# Now save the data
		for resp_name in data:
			fname = 'fig_sample_%s_%s.dat' % (samp_name, resp_name)
			pgf = PGF()
			pgf.add('N', Nvec[:i+1])
			p0, p25, p50, p75, p100 = np.percentile(data[resp_name][:i+1], [0, 25, 50, 75, 100], axis =0)
			pgf.add('p0', p0)
			pgf.add('p25', p25)
			pgf.add('p50', p50)
			pgf.add('p75', p75)
			pgf.add('p100', p100)
			pgf.write('data/'+fname)
