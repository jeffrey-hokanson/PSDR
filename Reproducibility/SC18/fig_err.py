import numpy as np
from psdr import PolynomialRidgeApproximation
from psdr.pgf import PGF
from multifDomains3D import buildDesignDomain, buildRandomDomain

design_domain = buildDesignDomain()
random_domain = buildRandomDomain(clip = 3)
total_domain = design_domain + random_domain

Xrand = np.loadtxt('rand2.input')
Xrand_norm = total_domain.normalize(Xrand)
Yrand = np.loadtxt('rand2_level11_v24.output')
Xdoe = np.loadtxt('stretch.input')
Xdoe_norm = total_domain.normalize(Xdoe)
Ydoe = np.loadtxt('stretch.output')

Xall_norm = np.vstack([Xrand_norm, Xdoe_norm[150:]])
Yall = np.vstack([Yrand, Ydoe[150:]])


ks = np.arange(150,800,2)
qois = [4,21,25]
#qois = [21,25]
for qoi in qois:
	Iall = np.isfinite(Yall[:,qoi])
	#norm = np.linalg.norm(Yall[Iall,qoi])
	norm = (np.nanmax(Yall[Iall,qoi]) - np.nanmin(Yall[Iall,qoi]))*np.sqrt(np.sum(Iall))
	err_rand_vec = []
	err_doe_vec = []
	for k in ks:
		I = np.isfinite(Yrand[:,qoi]) & (np.arange(Yrand.shape[0])<k)
		pra = PolynomialRidgeApproximation(degree = 3, subspace_dimension = 1, n_init = 1)
		pra.fit(Xrand_norm[I], Yrand[I,qoi])
		#err_rand = np.mean(np.abs(pra.predict(Xall_norm[Iall]) - Yall[Iall,qoi]))/norm
		err_rand = np.linalg.norm(pra.predict(Xall_norm[Iall]) - Yall[Iall,qoi])/norm
		err_rand_vec.append(err_rand)
		I = np.isfinite(Ydoe[:,qoi]) & (np.arange(Ydoe.shape[0])<k)
		pra.fit(Xdoe_norm[I], Ydoe[I,qoi])
		err_doe = np.linalg.norm(pra.predict(Xall_norm[Iall]) - Yall[Iall,qoi])/norm
		#err_doe = np.mean(np.abs(pra.predict(Xall_norm[Iall]) - Yall[Iall,qoi]))/norm
		err_doe_vec.append(err_doe)
		print "%4d: err rand %5.2e; doe %5.2e" % (k, err_rand, err_doe)

	pgf = PGF()
	pgf.add('k', ks)
	pgf.add('doe', err_doe_vec)
	pgf.add('rand', err_rand_vec)
	pgf.write('fig_err_qoi%d.dat' % (qoi,))

		
