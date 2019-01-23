from psdr.demos import GolinskiGearbox
from checkder import check_derivative

def test_grad():
	
	gb = GolinskiGearbox()
	
	for fun in [gb]:
		x = fun.sample(1)
		grad = lambda x: fun.grad(x).T
		assert check_derivative(x, fun, grad) < 1e-7	

