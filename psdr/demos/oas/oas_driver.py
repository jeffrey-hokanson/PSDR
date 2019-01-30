from __future__ import print_function
import numpy as np
from tqdm import tqdm
import argparse

from contextlib import contextmanager

# Make sure that OpenAeroStruct is in the PATH
import os, sys

sys.path.append('/root/Source/')

from OpenAeroStruct import OASProblem

# Constant determining the dimension of the parameterization
N_CP = 3

# Remove some of the verbosity
@contextmanager
def silence_stdout():
	new_target = open(os.devnull, "w")
	old_target, sys.stdout = sys.stdout, new_target
	try:
		yield new_target
	finally:
		sys.stdout = old_target

with silence_stdout():
	print("will not print")


null_stream = open(os.devnull, 'w')


def driver(X):

	assert len(X[0]) == (2*N_CP + 6), "Found dimension %d, expected %d" % (len(X[0]), (2*N_CP + 6))
	
   # Set problem type
	prob_dict = {'type' : 'aerostruct',
				'optimize' : False, # Don't optimize, only perform analysis
				'record_db' : False,
				'W0' : 1000., # OEW of the aircraft without fuel and the structural weight, in kg
				}

	# Instantiate problem and add default surface
	with silence_stdout():
		oas_prob = OASProblem(prob_dict)

	# Create a dictionary to store options about the surface.
	# Here we have 3 twist control points and 3 thickness control points.
	# We also set the discretization, span, and panel spacing.
	surf_dict = {'name' : 'wing',
				'symmetry' : True,
				'num_y' : 7,
				'num_x' : 2,
				'span' : 10.,
				'num_twist_cp' : N_CP,
				'num_thickness_cp' : N_CP,
				'num_chord_cp' : 2,
				'wing_type' : 'rect',
				'span_cos_spacing' : 0.,
				}

	# Add the specified wing surface to the problem
	with silence_stdout():
		oas_prob.add_surface(surf_dict)
	
	# Set up the problem. Once we have this done, we can modify the internal
	# unknowns and run the multidisciplinary analysis at that design point.
	oas_prob.add_desvar('wing.twist_cp')
	oas_prob.add_desvar('wing.thickness_cp')
	oas_prob.add_desvar('wing.chord_cp')
	oas_prob.setup(out_stream = null_stream)

	Y = np.zeros((X.shape[0], 5))
	for i, x in enumerate(tqdm(X)):
		twist_cp = x[:N_CP]
		thickness_cp = x[N_CP:2*N_CP]
		root_chord = x[2*N_CP]
		taper_ratio = x[2*N_CP+1]
		alpha = x[2*N_CP+2]
		E = x[2*N_CP+3]
		G = x[2*N_CP+4]
		mrho = x[2*N_CP+5]
	
		oas_prob.surfaces[0]['E'] = E
		oas_prob.surfaces[0]['G'] = G
		oas_prob.surfaces[0]['mrho'] = mrho
		with silence_stdout():
			oas_prob.setup(out_stream = null_stream)

		# set design variables
		oas_prob.prob['alpha'] = alpha
		oas_prob.prob['wing.twist_cp'] = twist_cp
		oas_prob.prob['wing.thickness_cp'] = thickness_cp
		oas_prob.prob['wing.chord_cp'] = np.array([taper_ratio, 1.]) * root_chord

		with silence_stdout():
			oas_prob.run()

		Y[i,0] = oas_prob.prob['fuelburn']
		Y[i,1] = oas_prob.prob['wing_perf.structural_weight']
		Y[i,2] = oas_prob.prob['wing_perf.L']
		Y[i,3] = oas_prob.prob['total_weight']
		Y[i,4] = oas_prob.prob['wing_perf.failure']

	return Y


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Run OpenAeroStruct')
	parser.add_argument(dest = 'infile', type = str)
	
	args = parser.parse_args()

	infile = args.infile
	X = np.loadtxt(infile)
	
	if len(X.shape) == 1:
		X = X.reshape(1,-1)

	Y = driver(X)	
	outfile = os.path.splitext(infile)[0] + '.output'

	np.savetxt(outfile, Y)

