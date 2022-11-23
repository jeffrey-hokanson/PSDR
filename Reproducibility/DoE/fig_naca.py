from __future__ import print_function, division
import numpy as np
import psdr, psdr.demos
from dask.distributed import Client, LocalCluster
from tqdm import tqdm
import matplotlib.pyplot as plt


if __name__ == '__main__':
	cluster = LocalCluster(processes = False, n_workers = 8)
	client = Client(cluster)
	fun = psdr.demos.NACA0012(dask_client = client)
	try:
		X = np.loadtxt('data/fig_naca_random.input')
	except IOError:
		np.random.seed(0)
		X = fun.domain.sample(1000)
		np.savetxt('data/fig_naca_random.input', X)

	try:
		fX = np.loadtxt('data/fig_naca_random.output')
	except IOError:
		res = fun.eval_async(X)
		fX = []
		for r in tqdm(res, ncols = 80):
			fX.append(r.result())
			np.savetxt('data/fig_naca_random.output', np.vstack(fX))

	lift = fX[:,0]
	drag = fX[:,1]

	pra_lift = psdr.PolynomialRidgeApproximation(subspace_dimension = 1, degree = 5)
	pra_lift.fit(X, lift)
	print(np.linalg.norm(pra_lift(X) - lift, np.inf)/(np.max(lift) - np.min(lift)))
	pra_drag = psdr.PolynomialRidgeApproximation(subspace_dimension = 1, degree = 5)
	pra_drag.fit(X, drag)
	print(np.linalg.norm(pra_drag(X) - drag, np.inf)/(np.max(drag) - np.min(drag)))

	if False:	
		fig, axes = plt.subplots(1,2)
		pra_lift.shadow_plot(X = X, fX = lift, ax = axes[0])
		axes[0].set_title('Lift')
		
		pra_drag.shadow_plot(X = X, fX = drag, ax = axes[1])
		axes[1].set_title('Drag')
		fig.tight_layout()	
		plt.show()
	#print(pra_lift.U)
	#plt.show()
