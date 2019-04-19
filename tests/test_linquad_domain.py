from __future__ import print_function
import numpy as np
from psdr import LinQuadDomain, BoxDomain


def test_isinside():
	m = 5
	L = np.eye(m)
	y = np.zeros(m)
	rho = 1

	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])
	
	X = np.random.randn(10,m)
	X = np.array([x/np.linalg.norm(x) for x in X])

	assert np.all(dom.isinside(0.5*X))
	assert np.all(dom.isinside(X))
	print(dom.isinside(1.1*X))
	assert np.all(~dom.isinside(1.1*X))


def test_extent_quad(m = 5):
	L = np.eye(m)
	y = np.random.randn(m)
	y *= 0.1/np.linalg.norm(y)
	rho = 1

	p = np.random.randn(m)

	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])
	x0 = np.random.randn(m)
	x0 *= 0.1/np.linalg.norm(x0)
	alpha = dom._extent_quad(x0, p)
	norm = np.linalg.norm(L.dot(x0 + alpha*p - y))
	print(alpha, norm, rho)
	assert np.isclose( norm, rho)

	# Now a pathological case where the direction is in the nullspace of the metric 
	L = np.zeros((1,m))
	L[0,0] = 1
	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])
	p = np.zeros(m)
	p[1] = 1
	
	alpha = dom._extent_quad(y, p)
	assert alpha == np.inf	


def test_corner(m = 5):
	L = np.eye(m)
	y = np.zeros(m)
	rho = 1

	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])

	p = np.random.randn(m)

	print(dom.norm_lb)
	print(dom.norm_ub)

	x = dom.corner(p)
	print(x)
	print(p/np.linalg.norm(p))
	assert np.all(np.isclose(x, p/np.linalg.norm(p)))


def test_closest_point(m = 5):
	L = np.eye(m)
	y = np.zeros(m)
	rho = 1

	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])

	x0 = np.random.randn(m)
	x0 *= 5/np.linalg.norm(x0)

	x = dom.closest_point(x0)
	print("x0", x0/np.linalg.norm(x0))
	print("x ", x)
	assert np.all(np.isclose(x, x0/np.linalg.norm(x0)))

	# Check when provided a L matrix	
	x = dom.closest_point(x0, L = L)
	print("x0", x0/np.linalg.norm(x0))
	print("x ", x)
	assert np.all(np.isclose(x, x0/np.linalg.norm(x0)))

def test_constrained_least_squares(m = 5):
	L = np.eye(m)
	y = np.zeros(m)
	rho = 1

	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])

	x0 = np.random.randn(m)
	x0 *= 5/np.linalg.norm(x0)

	x1 = dom.closest_point(x0)
	x2 = dom.constrained_least_squares(L, x0)
	print("x1", x1)
	print("x2", x2)
	assert np.all(np.isclose(x1,x2))


def test_sample(m = 5):
	L = np.eye(m)
	y = np.zeros(m)
	rho = 1

	dom = LinQuadDomain(Ls = [L], ys = [y], rhos = [rho])

	X = dom.sample(10)
	assert np.all(dom.isinside(X)), "sampler placed samples outside the domain"

def test_bad_scaling():
		
	# TODO: This fails with ub = 1e7
	# This is mainly due to solver tolerances
	lb = [-1, 1e7]
	ub = [1, 2e7]
	dom1 = BoxDomain(lb = lb, ub = ub)
	dom2 = LinQuadDomain(lb = lb, ub = ub, verbose = True)

	
	# Check quality of solution
	p = np.ones(len(dom1))
	# this calls an algebraic formula 
	x1 = dom1.corner(p)
	# whereas this calls a linear program
	x2 = dom2.corner(p)
	for x1_, x2_, lb_, ub_ in zip(x1, x2, dom2.lb, dom2.ub):
		print("x1:%+15.15e x2:%+15.15e delta:%+15.15e; lb: %+5.2e ub: %+5.2e" % (x1_, x2_, np.abs(x1_ - x2_), lb_, ub_))
	assert np.all(np.isclose(x1,x2))	
