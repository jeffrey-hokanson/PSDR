from __future__ import print_function
import numpy as np
import psdr

def test_pdf_full(m = 3):
	cov = np.eye(m)
	mean = np.zeros(m) 
	dom = psdr.NormalDomain(mean, cov)

	dom_box = psdr.BoxDomain(-10*np.ones(m), 10*np.ones(m))
	X, w = dom_box.quadrature_rule(1e6)
	p = np.sum(w*dom.pdf(X))
	print(p)
	assert np.isclose(p, 1.)


def test_pdf_truncate(m = 3):
	cov = np.eye(m)
	mean = np.zeros(m) 
	dom = psdr.NormalDomain(mean, cov, truncate = 5e-1)
	print(dom.norm_lb)
	dom_box = psdr.BoxDomain(dom.norm_lb, dom.norm_ub)
	X, w = dom_box.quadrature_rule(1e6)
	p = np.sum(w*dom.pdf(X))
	print(p)
	assert np.isclose(p, 1., rtol = 1e-3)
