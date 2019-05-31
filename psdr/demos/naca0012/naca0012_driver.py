import numpy as np
from SU2.io.config import Config
from SU2.io.tools import read_aerodynamics
import subprocess, os, argparse
import shutil
workdir = '/workdir'
su2home = os.environ['SU2_HOME']


def driver(x, n_lower = 10, n_upper = 10, maxiter = 1000):

	# Bumps exclude endpoints
	x_lower = np.linspace(0,1,n_lower+2)[1:-1]
	x_upper = np.linspace(0,1,n_upper+2)[1:-1]
	
	# Here we pull the demo config for inviscid NACA0012 included
	# in the SU2 QuickStart tutorial
	cfg = Config(os.path.join(su2home, 'QuickStart/inv_NACA0012.cfg'))
	#cfg = Config('inv_NACA0012.cfg')

	# Set the number of SU2 iterations
	cfg['EXT_ITER'] = maxiter
	
	# Set to use Hicks-Henne bump functions
	cfg['DV_MARKER'] = ['airfoil']
	cfg['DEFINITION_DV'] = {}
	cfg['DEFINITION_DV']['KIND'] = len(x)*['HICKS_HENNE']
	cfg['DEFINITION_DV']['SCALE'] = len(x)*[ 1.0 ]
	cfg['DEFINITION_DV']['MARKER'] = len(x)*[ ['airfoil'] ]
	cfg['DEFINITION_DV']['FFDTAG'] = len(x)*[ [ ] ]
	# lower surfaces, then upper 
	cfg['DEFINITION_DV']['PARAM'] =  [ [0, xi] for xi in x_lower] + [ [1, xi] for xi in x_upper] 
	
	# Set the deformation values
	cfg['DV_VALUE_NEW'] = x.tolist()

	# Specify the input and output for deformation
	shutil.copy(os.path.join(su2home, 'QuickStart/mesh_NACA0012_inv.su2'), 
		os.path.join(workdir, 'mesh_NACA0012_inv.su2'))
	cfg['MESH_FILENAME'] = 'mesh_NACA0012_inv.su2'
	cfg['MESH_OUT_FILENAME'] = 'mesh_deformed.su2'

	# Writes the config without commments
	cfg.dump(os.path.join(workdir, 'deform.cfg'))
   
	deformation_out_status = subprocess.run(['SU2_DEF','deform.cfg'],cwd=workdir)
	
	# Change which mesh gets loaded 
	cfg['MESH_FILENAME'] = 'mesh_deformed.su2'
	cfg['MESH_OUT_FILENAME'] = 'mesh_output.su2'
	cfg.dump(os.path.join(workdir, 'flow.cfg'))
	cfd_solver_out_status  = subprocess.run(['SU2_CFD','flow.cfg'],cwd=workdir)

	data = read_aerodynamics(os.path.join(workdir, 'history.dat'))
	print('lift', data['LIFT'])
	print('drag', data['DRAG'])
	return np.array([data['LIFT'], data['DRAG']])
	 

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description = 'Run SU2')
	parser.add_argument(dest = 'infile', type = str)
	parser.add_argument('--nlower', type=int, default=10) 
	parser.add_argument('--nupper', type=int, default=10) 
	args = parser.parse_args()

	# Load data
	infile = args.infile
	x = np.loadtxt(infile).flatten()
	n_lower = args.nlower
	n_upper = args.nupper
	
	y = driver(x, n_lower, n_upper)	
	outfile = os.path.splitext(infile)[0] + '.output'
	np.savetxt(outfile, y)
	
