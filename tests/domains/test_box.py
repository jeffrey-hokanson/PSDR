import numpy as np
import psdr

def test_unbounded(m = 5):
	dom = psdr.BoxDomain(-np.inf*np.ones(m), np.inf*np.ones(m))
	assert dom.is_unbounded == True
	
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	assert dom.is_unbounded == False

def test_point(m = 5):
	dom = psdr.BoxDomain(np.ones(m), np.ones(m))
	assert dom.is_point == True
	
	dom = psdr.BoxDomain(-np.ones(m), np.ones(m))
	assert dom.is_point == False
