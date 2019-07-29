import numpy as np
import psdr, psdr.demos
from psdr.pgf import PGF


if False:
	fun = psdr.demos.OTLCircuit()
	# Construct the testing dataset
	Xt = fun.domain.sample_grid(10)
	fun_name = 'otl'
elif True:
	fun = psdr.demos.Borehole()
	# Construct the testing dataset
	Xt = fun.domain.sample_grid(6)
	fun_name = 'borehole'

# Scaling for the test dataset
ft = fun(Xt).flatten()
scale = np.max(ft) - np.min(ft)

# Construct estimate of Lipschitz matrix
Xg = fun.domain.sample_grid(2)
grads = fun.grad(Xg)
lip = psdr.LipschitzMatrix()
lip.fit(grads = grads)
L = lip.L


samplers = {
	# should all take two arguments: the domain and the number of samples
	'random': psdr.random_sample,
	'maximin': lambda dom, N : psdr.maximin_sample(dom, N),
	'maximin_lip': lambda dom, N : psdr.maximin_sample(dom, N, L = lip.L),
#	'latin': lambda dom, N: psdr.latin_hypercube_maximin(dom, N),
#	'sobol': psdr.sobol_sequence, 
}

responses = {
	# Response surfaces to try out
	'pra_1_5': psdr.PolynomialRidgeApproximation(subspace_dimension = 1, degree = 5),
	'pra_2_5': psdr.PolynomialRidgeApproximation(subspace_dimension = 2, degree = 5),
#	'gp_iso': psdr.GaussianProcess(degree = 0),
#	'gp_iso_lin': psdr.GaussianProcess(degree = 1),
#	'gp_mat': psdr.GaussianProcess(structure = 'tril', degree = 0),
#	'gp_mat_lin': psdr.GaussianProcess(structure = 'tril', degree = 0),
#	'gp_lip': psdr.GaussianProcess(degree = 0, structure = 'scalar_mult', Lfixed = np.copy(lip.L)),  	
#	'gp_lip_lin': psdr.GaussianProcess(degree = 1, structure = 'scalar_mult', Lfixed = np.copy(lip.L)),  	
}


Nvec = [10,20,30,40,50,60,70,80,90,100]
Nvec = np.arange(10,1000+10, 10)
#Nvec = np.arange(6, 100, 2)
#Niter = 20
Niter = 1
for samp_name in samplers:
	samp = samplers[samp_name]
	print("Sampler: %s" % samp_name)	
	# Place to store the data
	data = {key: np.nan * np.zeros((len(Nvec), Niter)) for key in responses}

	for i, N in enumerate(Nvec):
		for it in range(Niter):
			# Construct sampling scheme
			np.random.seed(it)
			print("Building %4d point %15s design, iteration %3d" % (N, samp_name, it))
			X = samp(fun.domain, N)
			fX = fun(X).flatten()
			for resp_name in responses:	
				resp = responses[resp_name]
				# Now fit and test response surface
				try:
					resp.fit(X, fX)	
					# Record the sup norm error
					data[resp_name][i, it] = np.max(np.abs(resp(Xt).flatten() - ft.flatten() ))/scale
				except (KeyboardInterrupt, SystemExit):
					raise
				except:
					pass 

				print("\t err %10s: %8.2e" % (resp_name, data[resp_name][i,it]))

		# Now save the data
		for resp_name in data:
			fname = 'fig_sample_%s_%s_%s.dat' % (fun_name, samp_name, resp_name)
			pgf = PGF()
			pgf.add('N', Nvec[:i+1])
			p0, p25, p50, p75, p100 = np.nanpercentile(data[resp_name][:i+1,:], [0, 25, 50, 75, 100], axis =1)
			pgf.add('p0', p0)
			pgf.add('p25', p25)
			pgf.add('p50', p50)
			pgf.add('p75', p75)
			pgf.add('p100', p100)
			pgf.write('data/'+fname)
