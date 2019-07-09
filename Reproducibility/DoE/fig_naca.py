from __future__ import print_function, division
import numpy as np
import psdr, psdr.demos
from dask.distributed import Client, LocalCluster
from tqdm import tqdm

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

	res = fun.eval_async(X)
	fX = []
	for r in tqdm(res, ncols = 80):
		fX.append(r.result())
		np.savetxt('data/fig_naca_random.output', np.vstack(fX))
