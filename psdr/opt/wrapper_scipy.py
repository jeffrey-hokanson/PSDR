from shared import *
import numpy as np
from scipy.optimize import linprog 

def linprog_scipy(c, A_ub = None, b_ub = None, A_eq = None, b_eq = None,  lb = None, ub = None, eps = None, **kwargs):
	
	if lb is not None and ub is not None:
		bounds = [(lb_, ub_) for lb_, ub_ in zip(lb, ub)]
	elif ub is not None:
		bounds = [(None, ub_) for ub_ in ub]
	elif lb is not None:
		bounds = [(lb_, None) for lb_ in lb]
	else:
		bounds = None
	
	res = linprog(c, A_ub = A_ub, b_ub = b_ub, A_eq = A_eq, b_eq = b_eq, bounds = bounds, **kwargs)
	if res.success:
		return res.x
	else:
		raise Exception("Could not find feasible starting point: " + res.message)




