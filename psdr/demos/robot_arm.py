
import numpy as np

from psdr import BoxDomain

__all__ = ['build_robot_arm_domain', 'robot_arm']

def build_robot_arm_domain():
	# Parameters
	# theta 1-4, L 1-4
	lb = np.array([0,0,0,0,0,0,0,0])
	ub = np.array([2*np.pi, 2*np.pi, 2*np.pi, 2*np.pi, 1,1,1,1])
	return BoxDomain(lb, ub)

def robot_arm(X, return_grad = False):
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
	if not return_grad: return f
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
	return f, grad
