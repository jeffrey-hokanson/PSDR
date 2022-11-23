import numpy as np
import psdr 

def test_domain_type():
	dom = psdr.Domain()
	assert dom.is_linquad_domain == False
	assert dom.is_linineq_domain == False
	assert dom.is_box_domain == False
