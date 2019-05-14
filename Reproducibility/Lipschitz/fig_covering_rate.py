import numpy as np
from psdr.pgf import PGF

pgf = PGF()
pgf.read('data/fig_covering_mat.dat')
delta = np.array(pgf['eps'])
Ndelta = np.array(pgf['N'])

delta2 = 10.**( (np.log10(delta[1:]) + np.log10(delta[0:-1]))/2.)
slope2 = (np.log10(Ndelta[1:]) - np.log10(Ndelta[0:-1]))/(np.log10(delta[1:]) - np.log10(delta[0:-1]))

delta2 = delta2[np.isfinite(slope2)]
slope2 = slope2[np.isfinite(slope2)]

# Median filter 
slope_median = np.zeros(slope2.shape)
for i in range(len(slope2)):
	slope_median[i] = np.median(slope2[max(0, i-3):min(len(slope2),i+3)])

pgf = PGF()
pgf.add('eps', delta2)
pgf.add('slope', slope2)
pgf.add('median', slope_median)
pgf.write('data/fig_covering_mat_rate.dat')


if False:
	Iplus = slice(5,len(delta))
	Iminus = slice(0, len(delta)-5)
	delta5 = 10.**( (np.log10(delta[Iplus]) + np.log10(delta[Iminus]))/2.)
	slope5 = (np.log10(Ndelta[Iplus]) - np.log10(Ndelta[Iminus]))/(np.log10(delta[Iplus]) - np.log10(delta[Iminus]))

	pgf = PGF()
	pgf.add('delta', delta5)
	pgf.add('slope', slope5)
	pgf.write('data/fig_volume_slope5.dat')




