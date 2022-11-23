
import numpy as np

from psdr import BoxDomain, Function

__all__ = ['build_robot_arm_domain', 'robot_arm', 'RobotArm']

#TODO: Generalize to an arbitrary number of components
#TODO: compute derivative with complex finite-difference trick to avoid singularity 


class RobotArm(Function):
	r""" Robot Arm test function

	A test function that measures the distance of a four segment arm from the origin
	[VLSE_robot]_.

	.. math::

		f(\theta_1, \theta_2, \theta_3, \theta_4, L_1, L_2, L_3, L_4) &:=
			\sqrt{u^2 + v^2}, \text{ where }\\
			u &:= \sum_{i=1}^4 L_i \cos \left( \sum_{j=1}^i \theta_j\right) \\
			v &:= \sum_{i=1}^4 L_i \sin \left( \sum_{j=1}^i \theta_j\right)
	
	====================================    =============================
	Variable                                Interpretation
	====================================    =============================
	:math:`\theta_i \in [0, 2\pi]`          angle of the ith arm segment 
	:math:`L_i \in [0,1]`					length of the ith arm segment
	====================================    =============================

	Parameters
	----------
	dask_client: dask.distributed.Client or None
		If specified, allows distributed computation with this function.
	
	References
	----------
	.. [VLSE_robot] Virtual Library of Simulation Experiments, Robot Arm Function
		https://www.sfu.ca/~ssurjano/robot.html

	"""
	def __init__(self, dask_client = None):
		domain = build_robot_arm_domain()
		funs = [robot_arm]
		grads = [robot_arm_grad]
		
		Function.__init__(self, funs, domain, grads = grads, vectorized = True, dask_client = dask_client)

	def __str__(self):
		return "<Robot Arm Function>"

def build_robot_arm_domain():
	# Parameters
	# theta 1-4, L 1-4
	lb = np.array([0,0,0,0,0,0,0,0])
	ub = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1,1,1,1])
	return BoxDomain(lb, ub, names = ['theta_1', 'theta_2', 'theta_3', 'theta_4', 'L_1', 'L_2', 'L_3', 'L_4'])

def robot_arm(X):
	"""Robot arm test function
	
	See: https://www.sfu.ca/~ssurjano/robot.html
	"""
	X = X.reshape(-1, 8)

	th1 = X[:,0]
	th2 = X[:,1]
	th3 = X[:,2]
	th4 = X[:,3]
	L1  = X[:,4]
	L2  = X[:,5]
	L3  = X[:,6]
	L4  = X[:,7]

	u = L1*np.cos(th1) + L2*np.cos(th1+th2) + L3*np.cos(th1+th2+th3) + L4*np.cos(th1+th2+th3+th4)
	v = L1*np.sin(th1) + L2*np.sin(th1+th2) + L3*np.sin(th1+th2+th3) + L4*np.sin(th1+th2+th3+th4)

	f = np.sqrt(u**2 + v**2)
	return f

def robot_arm_grad(X):
	"""Robot arm test function
	
	See: https://www.sfu.ca/~ssurjano/robot.html
	"""
	X = X.reshape(-1, 8)

	th1 = X[:,0]
	th2 = X[:,1]
	th3 = X[:,2]
	th4 = X[:,3]
	L1  = X[:,4]
	L2  = X[:,5]
	L3  = X[:,6]
	L4  = X[:,7]

	u = L1*np.cos(th1) + L2*np.cos(th1+th2) + L3*np.cos(th1+th2+th3) + L4*np.cos(th1+th2+th3+th4)
	v = L1*np.sin(th1) + L2*np.sin(th1+th2) + L3*np.sin(th1+th2+th3) + L4*np.sin(th1+th2+th3+th4)

	f = np.sqrt(u**2 + v**2)
	grad = np.vstack([
			0*th1,
			-L1*(L2*np.sin(th2) + L3*np.sin(th2 + th3) + L4*np.sin(th2 + th3 + th4))/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
			-(L1*L3*np.sin(th2 + th3) + L1*L4*np.sin(th2 + th3 + th4) + L2*L3*np.sin(th3) + L2*L4*np.sin(th3 + th4))/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
			-L4*(L1*np.sin(th2 + th3 + th4) + L2*np.sin(th3 + th4) + L3*np.sin(th4))/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
			(L1 + L2*np.cos(th2) + L3*np.cos(th2 + th3) + L4*np.cos(th2 + th3 + th4))/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
			(L1*np.cos(th2) + L2 + L3*np.cos(th3) + L4*np.cos(th3 + th4))/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
			(L1*np.cos(th2 + th3) + L2*np.cos(th3) + L3 + L4*np.cos(th4))/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
			(L1*np.cos(th2 + th3 + th4) + L2*np.cos(th3 + th4) + L3*np.cos(th4) + L4)/np.sqrt(L1**2 + 2*L1*L2*np.cos(th2) + 2*L1*L3*np.cos(th2 + th3) + 2*L1*L4*np.cos(th2 + th3 + th4) + L2**2 + 2*L2*L3*np.cos(th3) + 2*L2*L4*np.cos(th3 + th4) + L3**2 + 2*L3*L4*np.cos(th4) + L4**2),
		]).T
	return grad
