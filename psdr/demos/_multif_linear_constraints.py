"""
Functions used in the construction of linear constraints for the 3-D nozzle
parameterization.

Rick Fenrich 2/26/18
"""
from __future__ import print_function

import numpy as np
from scipy import optimize

# General linear constraints which constrain a B-spline's motion by 1) preventing
# x-coordinates from swapping and 2) limiting slopes
# These have the form Ax <= b
# x = 1xN numpy array with equal number of x- and y-coordinates
# v = 1xN numpy array giving variable number for each control point
# throatIndex = integer giving Python index of throat location in x
# slopeLimits = (m_min_pre_throat, m_max_pre_throat, m_min_post_throat, m_max_post_throat)
#                NOTE: slope limits can be a list of numbers for each segment as well
# xLimits = (min_x_coord, max_x_coord)
# delta = small number for x-proximity of control points and y-proximity to throat
# throatIsLowest = flag (0 or 1) denoting whether to enforce throat is lowest constraint
# minThroat = value of minimum allowable radius in spline
# NOTE: if a throat is provided (throatIndex > 0), then the throat is assumed to 
#       be of the following form: a doubled control point at the throat, with 
#       a control point on either side of the throat which has been constrained
#       to have the same y-coordinate
def bspline(x, v, throatIndex, slopeLimits, xLimits=(None,None), delta=1e-2, 
            throatIsLowest=0, minThroat=None, maxThroat=None, output='verbose'):
    xleft = xLimits[0]; # minimum possible x-coordinate
    xright = xLimits[1]; # maximum possible x-coordinate
    delta2 = 1e-2; # for y-proximity to throat
    
    # Assume x is an 1xN numpy array with equal number of x- and y-coordinates
    n_local = len(x);
    nx = n_local//2;
    
    # Prepare linear constraint matrices, vectors, and counters
    B = np.zeros((100,n_local)); # temporary linear constraint matrix
    c = np.zeros((100,)); # temporary linear constraint vector
    m = 0; # number of rows in matrix (number of constraints)
    
    # Construct linear constraints for cross-over of x-coordinates
    m_initial = m;
    if xleft is not None:
        B[m,0] = -1; c[m] = -delta - xleft; m = m+1
    for i in range(1,nx):
        if i == throatIndex or i == throatIndex+1 or i == throatIndex+2:
            B[m,i] = -1; B[m,i-1] = 1; c[m] = -0.05; m = m+1
        else:
            B[m,i] = -1; B[m,i-1] = 1; c[m] = -delta; m = m+1
    if xright is not None:
        B[m,nx-1] = 1; c[m] = xright - delta; m = m+1
    if(output=='verbose'):
        print('constraints %i through %i used for cross-over of x-coord' % (m_initial+1, m));
 
    # Construct linear constraint for max width of throat
    # Used for throat of the type where 4 points share same coordinate; 
    # throat is doubled in middle with point on each side
    m_initial = m;
    if throatIndex != 0:
        B[m,throatIndex+2] = 1; B[m,throatIndex+1] = -1; c[m] = 0.1; m = m+1
        B[m,throatIndex] = 1; B[m,throatIndex-1] = -1; c[m] = 0.1; m = m+1
    if(output=='verbose'):
        print('constraints %i and %i used for max width of throat' % (m_initial+1,m_initial+2));

    if throatIsLowest == 1:
        # Construct linear constraints for throat is lowest point on left of throat
        if throatIndex != 0:
            m_initial = m;
            B[m,nx+throatIndex] = 1; B[m,nx+throatIndex-1] = -1; c[m] = -delta2; m = m+1
            if(output=='verbose'):
                print('constraints %i through %i used for throat is lowest on left' % (m_initial+1,m));
    
        # Construct linear constraints for throat is lowest point on right of throat
        if throatIndex != 0:
            m_initial = m;
            for i in range(nx+throatIndex+1,n_local):
                B[m,i] = -1; B[m,nx+throatIndex] = 1; c[m] = -delta2; m = m+1
            if(output=='verbose'):
                print('constraints %i through %i used for throat is lowest on right' % (m_initial+1,m));
            
    if minThroat is not None:
        m_initial = m;
        for i in range(nx,n_local):
            B[m,i] = -1; c[m] = -minThroat; m = m+1
        if(output=='verbose'):
            print('constraints %i through %i used for min throat radius constraint' %(m_initial+1,m));

    if maxThroat is not None:
        m_initial = m;
        for i in range(nx,n_local):
            B[m,i] = 1; c[m] = maxThroat; m = m+1
        if(output=='verbose'):
            print('constraints %i through %i used for max throat radius constraint' %(m_initial+1,m));
        
    # Set lower bound in slope segments prior to throat
    if throatIndex != 0 and slopeLimits[0] is not None:
        m_initial = m;
        n_seg = len(range(1,throatIndex)); # RWF used to be throatIndex+1
        if isinstance(slopeLimits[0],list):
            if len(slopeLimits[0]) == n_seg:
                m_bound = slopeLimits[0];
            else:
                print('Number of provided lower slope bounds in pre-throat is wrong (%i given, %i required)' % (len(slopeLimits[0]),n_seg));
                return 0, 0;
        else:
            m_bound = [slopeLimits[0]]*n_seg; # lower bound on normalized slope
        j = 0;
        for i in range(1,throatIndex): # RWF used to be throatIndex+1
            B[m,i] = m_bound[j]; B[m,i-1] = -m_bound[j]; B[m,i+nx] = -1; B[m,i+nx-1] = 1; c[m] = 0; m = m+1; j = j+1
        if(output=='verbose'):
            print('constraints %i through %i used for lower bound on slope segments prior' % (m_initial+1,m));

    # Set upper bound in slope segments prior to throat
    if throatIndex != 0 and slopeLimits[1] is not None:
        m_initial = m;
        n_seg = len(range(1,throatIndex)); # RWF used to be throatIndex+1
        if isinstance(slopeLimits[1],list):
            if len(slopeLimits[1]) == n_seg:
                m_bound = slopeLimits[1];
            else:
                print('Number of provided upper slope bounds in pre-throat is wrong (%i given, %i required)' % (len(slopeLimits[1]),n_seg));
                return 0, 0;
        else:
            m_bound = [slopeLimits[1]]*n_seg; # lower bound on normalized slope
        j = 0;
        for i in range(1,throatIndex): # RWF used to be throatIndex+1
            B[m,i] = -m_bound[j]; B[m,i-1] = m_bound[j]; B[m,i+nx] = 1; B[m,i+nx-1] = -1; c[m] = 0; m = m+1; j = j+1
        if(output=='verbose'):            
            print('constraints %i through %i used for upper bound on slope segments prior' % (m_initial+1,m));
        
    # Set lower bounds in slope segments after the throat
    if slopeLimits[2] is not None:
        m_initial = m;
        n_seg = len(range(throatIndex+2,nx-1)); # RWF used to be throatIndex
        if isinstance(slopeLimits[2],list):
            if len(slopeLimits[2]) == n_seg:
                m_bound = slopeLimits[2];
            else:
                print('Number of provided lower slope bounds in post-throat is wrong (%i given, %i required)' % (len(slopeLimits[2]),n_seg));
                return 0, 0;
        else:
            m_bound = [slopeLimits[2]]*n_seg; # lower bound on normalized slope  
         ## can make m_bound adaptive according to n_seg
        j = 0;
        for i in range(throatIndex+3,nx): # RWF used to be throatIndex+1
            B[m,i] = m_bound[j]; B[m,i-1] = -m_bound[j]; B[m,i+nx] = -1; B[m,i+nx-1] = 1; c[m] = 0; m = m+1; j = j+1
        if(output=='verbose'):  
            print('constraints %i through %i used for lower bound on slope segments after' % (m_initial+1,m));

    # Set upper bounds in slope segments after the throat
    if slopeLimits[3] is not None:
        m_initial = m;
        n_seg = len(range(throatIndex+2,nx-1)); # RWF used to be throatIndex
        if isinstance(slopeLimits[3],list):
            if len(slopeLimits[3]) == n_seg:
                m_bound = slopeLimits[3];
            else:
                print('Number of provided upper slope bounds in post-throat is wrong (%i given, %i required)' % (len(slopeLimits[3]),n_seg));
                return 0, 0;
        else:
            m_bound = [slopeLimits[3]]*n_seg; # upper bound on normalized slope
        ## can make m_bound adaptive according to n_seg
        j = 0;
        for i in range(throatIndex+3,nx): # RWF used to be throatIndex+1
            B[m,i] = -m_bound[j]; B[m,i-1] = m_bound[j]; B[m,i+nx] = 1; B[m,i+nx-1] = -1; c[m] = 0; m = m+1; j = j+1
        if(output=='verbose'):
            print('constraints %i through %i used for upper bound on slope segments after' % (m_initial+1,m));
                
    # Remove columns from B that are not design variables
    A = np.zeros((m,max(v)));
    b = np.zeros((m,));
    b[:] = c[:m];
    for i in range(n_local):
        if v[i] == 0: # x_i is not a design variable
            b[:] = b[:] - x[i]*B[:m,i];
        else:
            A[:,v[i]-1] = A[:,v[i]-1] + B[:m,i];
            
    if(output=='verbose'):
        print();
            
    return A, b


# General linear constraints which constrain a piecewise bilinear shape by 
# 1) preventing x-coordinates from swapping and 2) limiting slopes
# These constraints have the form Ax <= b
# x = 1xN numpy array with values in direction 1
# y = 1xM numpy array with values in direction 2
# z = z(x,y) = 1x(NxM) numpy array with function values for x and y ordered 
#     such that x is cycled through for every y: (x1, y1) then (x2, y1) ...
# v = 1xN numpy array giving variable number for each control point
# slopeLimits = (m_min_dir1, m_max_dir2)
# xLimits = (min_x_coord, max_x_coord)
# deltax = small number for x-proximity of break locations
# deltay = small number for y-proximity of break locations
# deltaz = small number for z-proximity of break locations in radial dir only
def piecewiseBilinearAxial(x, y, z, v, slopeLimits, xLimits=(None,None), 
                      deltax=1e-2, deltay=1e-2, deltaz = 1e-2,
                      output = 'verbose'):
                          
    xleft = xLimits[0]; # minimum possible x-coordinate
    xright = xLimits[1]; # maximum possible x-coordinate
    m_minx = slopeLimits[0];
    m_maxx = slopeLimits[1];
    
    nx = len(x);
    ny = len(y);
    nz = len(z);
    n_local = len(v);
    if( nz != nx*ny ):
        raise RuntimeError('Incorrect number of provide function values ' +
        '(%i provided instead of %i).\n' % (nz,nx*ny));
    elif( n_local != nx + ny + nz ):
        raise RuntimeError('Incorrect number of provided design variable ' +
        'indices (%i provided instead of %i).\n' % (n_local, nx+ny+nz));
    
    # Prepare linear constraint matrices, vectors, and counters
    B = np.zeros((100,n_local)); # temporary linear constraint matrix
    c = np.zeros((100,)); # temporary linear constraint vector
    m = 0; # number of rows in matrix (number of constraints)  
    
    # Construct linear constraints for cross-over of x-coordinates
    m_initial = m;
    if xleft is not None:
        B[m,0] = -1; c[m] = -deltax - xleft; m = m+1
    for i in range(1,nx):
        B[m,i] = -1; B[m,i-1] = 1; c[m] = -deltax; m = m+1
    if xright is not None:
        B[m,nx-1] = 1; c[m] = xright - deltax; m = m+1
    if(output=='verbose'):
        print('constraints %i through %i used for x-proximity' % (m_initial+1, m));    

    # Construct linear constraints for cross-over of y-coordinates
    if( deltay is not None ):
        m_initial = m;
        for i in range(1+nx-1,ny+nx-1):
            B[m,i] = -1; B[m,i-1] = 1; c[m] = -deltay; m = m+1
        if(output=='verbose'):
            print('constraints %i through %i used for y-proximity' % (m_initial+1, m));
        
    # Set lower bounds on slopes in x-direction
    m_initial = m;
    for i in range(1,nx):
        for j in range(0,ny):
            B[m,i] = m_minx; B[m,i-1] = -m_minx; B[m,nx+ny+j*nx+i-1] = 1; B[m,nx+ny+j*nx+i] = -1; c[m] = 0; m = m+1
    if(output=='verbose'):
        print('constraints %i through %i used for lower bound on slope segments in x-direction' % (m_initial+1, m));    

    # Set upper bounds on slopes in x-direction
    m_initial = m;
    for i in range(1,nx):
        for j in range(0,ny):
            B[m,i] = -m_maxx; B[m,i-1] = m_maxx; B[m,nx+ny+j*nx+i-1] = -1; B[m,nx+ny+j*nx+i] = 1; c[m] = 0; m = m+1
    if(output=='verbose'):
        print('constraints %i through %i used for upper bound on slope segments in x-direction' % (m_initial+1, m));    

    # Set thickness proximity in y-direction
    if( deltaz is not None ):
        m_initial = m;
        for i in range(nx):
            k = nx + ny + i;
            k2 = nx + ny + i +(ny-1)*nx;
            B[m,k] = 1; B[m,k2] = -1; c[m] = deltaz; m = m+1
            B[m,k] = -1; B[m,k2] = 1; c[m] = deltaz; m = m+1
            for j in range(1,ny):
                k = nx + ny + (j-1)*nx + i;
                k2 = nx + ny + j*nx + i;
                B[m,k2] = 1; B[m,k] = -1; c[m] = deltaz; m = m+1
                B[m,k2] = -1; B[m,k] = 1; c[m] = deltaz; m = m+1
        if(output=='verbose'):
            print('constraints %i through %i used for z proximity in y-direction' % (m_initial+1, m));    

    # Remove columns from B that are not design variables
    A = np.zeros((m,max(v)));
    b = np.zeros((m,));
    b[:] = c[:m];
    xtmp = np.hstack((x,y,z));
    for i in range(n_local):
        if v[i] == 0: # x_i is not a design variable
            b[:] = b[:] - xtmp[i]*B[:m,i];
        else:
            A[:,v[i]-1] = A[:,v[i]-1] + B[:m,i];
            
    if(output=='verbose'):
        print();
    
    return A, b

    
# Baffle linear constraints which constrain a monotonically increasing vector 
# x such that each element is spaced between minDistance and maxDistance apart
# These constraints have the form Ax <= b
# x = 1xN numpy array with baffle locations (used)
# t = 1xN numpy array with baffle thicknesses (not used)
# h = float for baffle transverse extent (the amount the baffle extends in 
#     transverse direction)
# v = 1xN numpy array giving variable number for each control point
# minDistance = minimum distance baffles can be together
# maxDistance = maximum distance baffles can be apart
def baffles(x, t, h, v, minDistance, maxDistance, output='verbose'):

    n = (len(x)-1)*2; # number of linear constraints
    nx = len(x); # length of baffle locations vector
    nv = len(v); # length of variables
    B = np.zeros((n, nv)); # placeholder matrix for A in Ax <= b
    c = np.zeros(n); # placeholder vectory for b in Ax <= b
    
    m = 0;
    # Set constraints on x-position
    m_initial = m;
    for i in range(1,nx):
        B[m,i] = 1; B[m,i-1] = -1; c[m] = maxDistance; m += 1
        B[m,i] = -1; B[m,i-1] = 1; c[m] = -minDistance; m += 1
    if(output=='verbose'):
        print('constraints %i through %i used for baffle proximities' % (m_initial+1, m));          

    # Remove columns from B that are not design variables
    A = np.zeros((m,max(v)));
    b = np.zeros((m,));
    b[:] = c[:m];
    xtmp = np.hstack((x,t));
    for i in range(nv):
        if v[i] == 0: # x_i is not a design variable
            b[:] = b[:] - xtmp[i]*B[:m,i];
        else:
            A[:,v[i]-1] = A[:,v[i]-1] + B[:m,i];
            
    if(output=='verbose'):
        print();
    
    return A, b
    

# This function is used remove unecessary rows from the constraint matrix. It 
# is also used to concatenate linear constraint matrices for domains which 
# share the same variables, as in the 3 B-splines for the 3-D inner wall 
# parameterization.    
def cleanupConstraintMatrix(Alist=[],blist=[]):
    
    # For each provided constraint matrix
    mlist = [];
    nlist = [];
    for i in range(len(Alist)):        
        m,n = Alist[i].shape;
        mlist.append(m);
        nlist.append(n);
        
    # Concatenate the arrays (assumes design variables are the same)
    mtot = np.sum(mlist);
    ndv = np.max(nlist); # all n should be the same
    A = np.zeros((mtot,ndv));
    b = np.zeros((mtot,1));
    m1 = 0;
    for i in range(len(Alist)):
        m2 = m1 + mlist[i];
        n2 = nlist[i];
        A[m1:m2,0:n2] = Alist[i];
        b[m1:m2,0] = blist[i];
        m1 = m2;
        
    m, n = A.shape;
        
    # Cleanup constraints by eliminating unnecessary rows
    rowsToDelete = [];
    for i in range(m):
        if np.count_nonzero(A[i,:]) == 0: # no entries
            rowsToDelete.append(i);
    
    A = np.delete(A,np.array(rowsToDelete, dtype = np.int),0)
    b = np.delete(b,np.array(rowsToDelete, dtype = np.int),0)
    
    return A, b;


# A hit and run method for sampling from a polytope.    
#def hitAndRun(N, A, b, lb, ub):
#
#    m, n = A.shape
#        
#    # Attempt to find initial feasible point at Chebyshev center
#    normA = np.sqrt( np.sum( np.power(A, 2), axis=1 ) ).reshape((m, 1))
#    AA = np.hstack(( A, normA ))
#    c = np.zeros((n+1,))
#    c[-1] = -1.0
#    if lb.size < 1: # do not use bounds
#        result = optimize.linprog(c,A_ub=AA,b_ub=b)
#        #print result
#        z0 = result.x[0:-1].reshape((n,1))   
#    else:
#        raise NotImplementedError('Bounds are not currently implemented.');
#        bounds_list = []
#        for i in range(x.n):
#            bounds_list.append((lb[i],ub[i]))
#        bounds_list.append((None,None))
#        result = optimize.linprog(c,A_ub=AA,b_ub=b,bounds=tuple(bounds_list))
#        z0 = result.x[0:-1].reshape((n,1))
#    
#    # Reshape necessary arrays
#    b = b.reshape((m,1))
#
#    # tolerance
#    ztol = 1e-6
#    eps0 = ztol/4.0
#
#    Z = np.zeros((N, n))
#    for i in range(N):
#
#        # random direction
#        bad_dir = True
#        count, maxcount = 0, int(1e6)
#        while bad_dir:
#            d = np.random.normal(size=(n, 1))
#            bad_dir = np.any(np.dot(A, z0 + eps0*z0*d/np.linalg.norm(d)) > b)
#            count += 1
#            if count >= maxcount:
#                print 'hitAndRun error: reached max bad direction count'
#                Z[i:,:] = np.tile(z0, (1,N-i)).transpose()
#                return -1
#
#        # find constraints that impose lower and upper bounds on eps
#        f, g = b - np.dot(A,z0), np.dot(A, d)
#
#        # find an upper bound on the step
#        min_ind = np.logical_and(g>=0, f > np.sqrt(np.finfo(np.float).eps))
#        eps_max = np.amin(f[min_ind]/g[min_ind])
#
#        # find a lower bound on the step
#        max_ind = np.logical_and(g<0, f > np.sqrt(np.finfo(np.float).eps))
#        eps_min = np.amax(f[max_ind]/g[max_ind])
#
#        # randomly sample eps
#        eps1 = np.random.uniform(eps_min, eps_max)
#
#        # take a step along d
#        z1 = z0 + eps1*d
#        Z[i,:] = z1.reshape((n, ))
#
#        # update temp var
#        z0 = z1.copy()
#
#    return Z
