import numpy as np
from domains import Domain, UnboundedDomain
from gn import trajectory_linear, linesearch_armijo


__all__ = ['minimax',
	]

def minimax(f, x0, domain = None, linesearch = linesearch_armijo,
	trajectory = trajectory_linear, maxiter = 100):
	r""" Solves a minimax optimization problem via sequential linearization and line search

	Given a vector-valued function :math:`f: \mathcal{D}\subset\mathbb{R}^m\to \mathbb{R}^q`,
	solve the minimax optimization problem:

	.. math::
	
		\min_{\mathbf{x}\in \mathcal{D}}  \max_{i=1,\ldots, q} f_i(\mathbf{x}).

	This implements a minor modification of the algorithm described by Osborne and Watson [OW69]_.
	The basic premise is that we add in a slack variable :math:`t_k` representing the maximum value
	of :math:`f_i(\mathbf{x}_k)` at the current iterate :math:`\mathbf{x}_k`.
	Then we linearize the functions :math:`f_i` about this point, 
	yielding additional linear constraints: 
	

	.. math::
		:label: minimax1 

		\min_{\mathbf{p}_\mathbf{x} \in \mathbb{R}^m, p_t}  & \  p_t \\
		\text{such that} & \ t_k + p_t \ge f_i(\mathbf{x}_k) + 
			\mathbf{p}_\mathbf{x}^\top \nabla f_i(\mathbf{x}_k) \quad \forall i=1,\ldots,q

	This yields a search direction :math:`\mathbf{p}_\mathbf{x}`,
	along which we preform a line search to find a point satisfying the constraint:

	.. math::
		:label: minimax2

		\min_{\alpha \ge 0} \max_{i=1,\ldots,q} f_i(T(\mathbf{x}_k, \mathbf{p}_\mathbf{x}, \alpha))
			\le t_k + \alpha p_t.

	Here :math:`T` represents the trajectory which defaults to a linear trajectory:

	.. math::
		
		T(\mathbf{x}, \mathbf{p}, \alpha) = \mathbf{x} + \alpha\mathbf{p}

	but, if provided can be more sophisticated. 
	The substantial difference from [OW69]_ is using an inexact backtracking linesearch
	is used to find the :math:`\alpha` satisfying the Armijo like condition :eq:`minimax2`;
	as originally proposed, Osborne and Watson use an exact line search. 

	Parameters
	----------
	f : callable
		


	
	References
	----------
	.. [OW69] Osborne and Watson, "An algorithm for minimax approximation in the nonlinear case",
		The Computer Journal, 12(1) 1969 pp. 63--68
		https://doi.org/10.1093/comjnl/12.1.63

	"""

	x0 = np.array(x0)

	if domain is None:
		domain = UnboundedDomain(len(x0))

	assert isinstance(domain, Domain), "Must provide a domain for the space"
	assert domain.isinside(x0), "Starting point must be inside the domain"


	# Start of optimization loop
	for it in range(maxiter):
		fx = f(x0)



if __name__ == '__main__':
	x0 = np.random.randn(5)
	f = lambda x: np.abs(x)
	minimax(None, x0)
