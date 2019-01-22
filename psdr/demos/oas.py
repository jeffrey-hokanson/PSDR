import numpy as np

from psdr import BoxDomain, TensorProductDomain, UniformDomain


def build_oas_design_domain(n_cp = 3):
	# Twist
	domain_twist = BoxDomain(-1*np.ones(n_cp), 1*np.ones(n_cp))
	# Thick
	domain_thick = BoxDomain(0.005*np.ones(n_cp), 0.05*np.ones(n_cp))
	# Root Chord
	domain_root_chord = BoxDomain(0.7, 1.3)
	# Taper ratio
	domain_taper_ratio = BoxDomain(0.75, 1.25)

	return TensorProductDomain([domain_twist, domain_thick, domain_root_chord, domain_taper_ratio])

def build_oas_robust_domain():
	# alpha - Angle of Attack
	return BoxDomain(2.0, 5.0)

def build_oas_random_domain():
	E = UniformDomain(0.8*70e9, 1.2*70e9)	 
	G = UniformDomain(0.8*30e9, 1.2*30e9)
	rho = UniformDomain(0.8*3e3, 1.2*3e3)
	return TensorProductDomain([E,G,rho])


def oas(x, version = 'v1'):
	import numpy as np
	pass

