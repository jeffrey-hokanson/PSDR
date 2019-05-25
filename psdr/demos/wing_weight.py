import numpy as np

from psdr import BoxDomain, Function


__all__ = ['build_wing_weight_domain' ,'wing_weight', 'wing_weight_grad', 'WingWeight']


class WingWeight(Function):
	r""" The wing weight test function

	This function models the weight of a wing based on several design parameters [VLSE_wing]_:

	.. math::
		
		&f(S_w, W_{fw}, A, \Lambda, q, \lambda, t_c, N_z, W_{dg}, W_p) := \\
			&\quad 0.036 S_w^{0.758} W_{fw}^{0.0035}
			\left( \frac{A}{\cos^2 \Lambda}  \right)^{0.6} 
			q^{0.006} \lambda^{0.04}
			\left( \frac{100 t_c}{\cos \Lambda} \right)^{-0.3}
			(N_z W_{dg})^{0.49} 
			+ S_w W_p
	
	====================================    ========================
	Variable                                Interpretation
	====================================    ========================
	:math:`S_w \in [150, 200]`				wing area (ft^2)
	:math:`W_{fw} \in [220, 300]`			weight of fuel in the wing (lb)
	:math:`A \in [6,10]`					aspect ratio
	:math:`\Lambda \in [-10,10]`			quarter-chord sweep (degrees)
	:math:`q \in [16,45]`					dynamic pressure at cruise (lb/ft^2)
	:math:`\lambda \in [0.5,1]`				taper ratio
	:math:`t_c\in [0.08,0.18]`				aerfoil thickness to chord ratio
	:math:`N_z \in [2.5, 6]`				ultimate load factor
	:math:`W_{dg} \in [1700, 2500]`			flight design gross weight (lb)
	:math:`W_p \in [0.025, 0.08]`			paint weight (lb/ft^2)
	====================================    ========================

	References
	----------
	.. [VLSE_wing] Virtual Library of Simulation Experiments, Wing Weight Function
		 https://www.sfu.ca/~ssurjano/wingweight.html


	"""
	def __init__(self):
		domain = build_wing_weight_domain()
		funs = [wing_weight]
		grads = [wing_weight_grad]
		Function.__init__(self, funs, domain, grads = grads, vectorized = True)	

	def __str__(self):
		return "<Wing Weight Function>"

def build_wing_weight_domain():
	# Variables
	# Sw, Wfw, A, Lam, q, lam, tc, Nz, Wdg, Wp
	lb = np.array([150, 220, 6, -10, 16, 0.5, 0.08, 2.5, 1700, 0.025])
	ub = np.array([200, 300, 10, 10, 45,   1, 0.18,   6, 2500, 0.08])
	return BoxDomain(lb, ub, names = ['S_w', 'W_fw', 'A', 'Lambda', 'q', 'lambda', 't_c', 'N_z', 'W_dg', 'W_p'])



def wing_weight(X):
	"""Wing weight test function
	
	See: https://www.sfu.ca/~ssurjano/wingweight.html
	"""	

	X = X.reshape(-1,10)

	Sw  = X[:,0]
	Wfw = X[:,1]
	A   = X[:,2]
	Lam = X[:,3]*(np.pi/180) # In degrees
	q   = X[:,4]
	lam = X[:,5]
	tc  = X[:,6]
	Nz  = X[:,7]
	Wdg = X[:,8]
	Wp  = X[:,9]

	f = 0.036*(Sw**0.758)*(Wfw**0.0035)*(A/(np.cos(Lam)**2))**(0.6)*(q**0.006)*(lam**0.04)*(100*tc/np.cos(Lam))**(-0.3)*(Nz*Wdg)**(0.49) + Sw*Wp
	return f

def wing_weight_grad(X):
	X = X.reshape(-1,10)

	Sw  = X[:,0]
	Wfw = X[:,1]
	A   = X[:,2]
	Lam = X[:,3]*(np.pi/180) # In degrees
	q   = X[:,4]
	lam = X[:,5]
	tc  = X[:,6]
	Nz  = X[:,7]
	Wdg = X[:,8]
	Wp  = X[:,9]
		
	grad = np.vstack([
			0.00685443569430334*Sw**(-0.242)*Wfw**0.0035*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3) + 1.0*Wp,
			3.16497690370207e-5*Sw**0.758*Wfw**(-0.9965)*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3),
			0.00542567469206069*Sw**0.758*Wfw**0.0035*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3)/A,
			(np.pi/180)*0.00813851203809104*Sw**0.758*Wfw**0.0035*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3)*np.tan(Lam),
			5.42567469206069e-5*Sw**0.758*Wfw**0.0035*lam**0.04*q**(-0.994)*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3),
			0.000361711646137379*Sw**0.758*Wfw**0.0035*lam**(-0.96)*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3),
			-0.00271283734603035*Sw**0.758*Wfw**0.0035*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3)/tc,
			0.0044309676651829*Sw**0.758*Wfw**0.0035*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3)/Nz,
			0.0044309676651829*Sw**0.758*Wfw**0.0035*lam**0.04*q**0.006*(A/np.cos(Lam)**2)**0.6*(Nz*Wdg)**0.49*(tc/np.cos(Lam))**(-0.3)/Wdg,
			Sw,
		]).T
	return grad	
