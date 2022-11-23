import numpy as np
from psdr.pgf import PGF

pgf = PGF()
pgf.read('data/fig_covering_mat.dat')
eps = np.array(pgf['eps'])
Ns = np.array(pgf['N'])


slope = np.zeros(eps.shape)
for i in range(len(slope)):
	I = slice(max(0,i-1), min(len(slope), i+1))
	xx = np.log10(eps[I])
	yy = np.log10(Ns[I])
	# Fit a line
	p = np.polyfit(xx, yy, 1)
	slope[i] = p[0]

# Median filter 
slope_median = np.zeros(slope.shape)
for i in range(len(slope)):
	slope_median[i] = np.median(slope[max(0, i-3):min(len(slope),i+3)])

pgf = PGF()
pgf.add('eps', eps)
pgf.add('slope', slope)
pgf.add('median', slope_median)
pgf.write('data/fig_covering_mat_slope.dat')


