import numpy as np
from SU2.io.config import Config
from SU2.io.tools import read_aerodynamics
import subprocess, os


workdir = '/workdir'
su2home = os.environ['SU2_HOME']

if __name__ == '__main__':
    x = -0.1*np.ones(20)
    n_lower = 10
    n_upper = 10
    maxiter = 500

    # Bumps exclude endpoints
    x_lower = np.linspace(0,1,n_lower+2)[1:-1]
    x_upper = np.linspace(0,1,n_upper+2)[1:-1]
    
    cfg = Config('NACA0012.cfg')
    #cfg = Config(os.path.join(su2home, 'QuickStart/inv_NACA0012.cfg'))
    # Set the number of SU2 iterations
    cfg['EXT_ITER'] = maxiter
    #cfg['LINEAR_SOLVER_PREC'] = 'JACOBI'
    #cfg['CONV_NUM_METHOD_FLOW'] = 'LAX-FRIEDRICH'
    #cfg['CONV_NUM_METHOD_FLOW'] = 'ROE'
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

    data = read_aerodynamics(os.path.join(workdir, 'history.csv'))
    print('lift', data['LIFT'])
    print('drag', data['DRAG'])
     

