from __future__ import print_function
from psdr.demos import GolinskiGearbox, Borehole, OTLCircuit, Piston, RobotArm, WingWeight
from checkder import check_derivative

def test_grad():
	
	borehole = Borehole()
	gb = GolinskiGearbox()
	otl = OTLCircuit()
	piston = Piston()
	robotarm = RobotArm()
	wing = WingWeight()
	
	for fun in [borehole, gb, otl, piston, robotarm, wing]:
		x = fun.sample(1)
		grad = lambda x: fun.grad(x).T
		print(fun(x))
		print(grad(x))
		assert check_derivative(x, fun, grad) < 1e-7	
