import numpy as np

from ._multif_linear_constraints import bspline
from ._multif_linear_constraints import piecewiseBilinearAxial
from ._multif_linear_constraints import baffles
from ._multif_linear_constraints import cleanupConstraintMatrix

from psdr import LinIneqDomain, UniformDomain, LogNormalDomain, TensorProductDomain

def buildDesignDomain(output='verbose', **kwargs):

	lb_perc = 0.8
	ub_perc = 1.2

	# ============================================================================
	# Specify design variables
	# ============================================================================
	# The choice of design variables below has 4 control points which control the 
	# nozzle throat. Control points are duplicated at the throat, and there is one
	# on either side of the throat which helps give a smooth (and not pointed)
	# throat. The centerline has a throat as well which coincides with the major
	# axis throat. There is no throat for the minor axis. Instead, the minor axis
	# monotonically decreases.

	# ============================================================================
	# Wall design variables for free inlet
	# ============================================================================
	# # Centerline
	# WALL_COEFS1 = (0.0000, 0.0000, 0.3000, 0.5750, 1.1477, 1.1500, 1.1500, 1.1523, 1.7262, 2.0000, 2.3000, 2.3000, 
	#				0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000)
	# WALL_COEFS1_DV= (1,	1,	  2,	  3,	  4,	  5,	  5,	  6,	  7,	  0,	  0,	  0,	 
	#				  8,	8,	  8,	  9,	  10,	 10,	 10,	 10,	 11,	 0,	  0,	  0)
	
	# # Major Axis
	# WALL_COEFS2= (0.0000, 0.0000, 0.3000, 0.5000, 0.7000, 0.9000, 1.1477, 1.1500, 
	#			   1.1500, 1.1523, 1.4000, 1.6500, 1.9000, 2.1000, 2.3000, 2.3000, 
	#			   0.3255, 0.3255, 0.3255, 0.3195, 0.3046, 0.2971, 0.2956, 0.2956, 
	#			   0.2956, 0.2956, 0.3065, 0.3283, 0.3611, 0.4211, 0.4265, 0.4265)
	# WALL_COEFS2_DV= (1,   1,	  2,	  12,	 13,	 14,	 15,	 5,
	#				  5,   16,	 17,	 18,	 19,	 20,	 0,	  0,
	#				  0,   0,	  0,	  21,	 22,	 23,	 24,	 24,
	#				  24,  24,	 25,	 26,	 27,	 28,	 0,	  0)
					 
	# # Minor axis
	# WALL_COEFS3= (0.0000, 0.0000, 0.3000, 0.5500, 0.9000, 
	#			   1.1500, 1.8000, 2.1000, 2.3000, 2.3000, 
	#			   0.3255, 0.3255, 0.3255, 0.3195, 0.2956, 
	#			   0.2750, 0.2338, 0.2167, 0.2133, 0.2133)
	# WALL_COEFS3_DV= (1,   1,	  2,	  29,	 30,
	#				  31,  32,	 33,	 0,	  0, 
	#				  0,   0,	  0,	  34,	 35,  
	#				  36,  37,	 38,	 0,	  0)

	# ============================================================================
	# Wall design variables for fixed inlet
	# ============================================================================
	# Wall parameterization (centerline, major axis, minor axis, shovel exit height, and inlet angle)
	WALL_COEFS1= (0.0000, 0.0000, 0.3000, 0.5750, 1.1500, 1.7262, 2.0000, 2.33702, 2.33702, 0.099908, 0.099908, 0.099908, 0.12, 0.14, 0.17, 0.19, 0.19, 0.19)
	WALL_COEFS1_DV= (0, 0, 0, 1, 2, 3, 0, 0, 0, 0, 0, 0, 4, 5, 6, 0, 0, 0)
	WALL_COEFS2= (0.0000, 0.0000, 0.3000, 0.7000, 1.1500, 1.6000, 1.8, 2.33702, 2.33702, 0.439461, 0.439461, 0.439461, 0.6, 0.7, 0.8, 0.85, 0.92, 0.92)
	WALL_COEFS2_DV= (0, 0, 0, 7, 8, 9, 10, 0, 0, 0, 0, 0, 11, 12, 13, 14, 0, 0)
	WALL_COEFS3= (0.0000, 0.0000, 0.3000, 0.7000, 1.1500, 1.6000, 2.33702, 2.33702, 0.439461, 0.439461, 0.439461, 0.3, 0.29, 0.26, 0.24, 0.24)
	WALL_COEFS3_DV= (0, 0, 0, 7, 8, 15, 0, 0, 0, 0, 0, 16, 17, 18, 0, 0)	
	WALL_SHOVEL_HEIGHT= -0.1
	WALL_SHOVEL_START_ANGLE= 20

	# ---- LAYER THICKNESS GEOMETRIC PARAMETERIZATION ----

	# Inner thermal layer takes the heat load
	# LAYER1= (THERMAL_LAYER, PIECEWISE_BILINEAR, CMC)
	LAYER1_THICKNESS_LOCATIONS= (0, 0.5, 1.0)
	LAYER1_THICKNESS_ANGLES= (0, 90, 180, 270)
	LAYER1_THICKNESS_VALUES= (0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03)
	LAYER1_DV= (0, 1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

	# Air gap between thermal and load layers
	# LAYER2= (AIR_GAP, CONSTANT, AIR)
	LAYER2_THICKNESS = 0.0254

	# Lower layer of load layer (Gr/BMI composite material)
	# LAYER3= (LOAD_LAYER_INSIDE, PIECEWISE_BILINEAR, GR-BMI)
	LAYER3_THICKNESS_LOCATIONS= (0, 0.5, 1.0)
	LAYER3_THICKNESS_ANGLES= (0, 90, 180, 270)
	LAYER3_THICKNESS_VALUES= (0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002)
	LAYER3_DV= (0, 1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

	# Middle layer of load layer (Ti-honeycomb)
	# LAYER4= (LOAD_LAYER_MIDDLE, PIECEWISE_BILINEAR, GR-BMI)
	LAYER4_THICKNESS_LOCATIONS= (0, 0.5, 1.0)
	LAYER4_THICKNESS_ANGLES= (0, 90, 180, 270)
	LAYER4_THICKNESS_VALUES= (0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.013)
	LAYER4_DV= (0, 1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

	# Upper layer of load layer (Gr/BMI composite material)
	# LAYER5= (LOAD_LAYER_OUTSIDE, PIECEWISE_BILINEAR, GR-BMI)
	LAYER5_THICKNESS_LOCATIONS= (0, 0.5, 1.0)
	LAYER5_THICKNESS_ANGLES= (0, 90, 180, 270)
	LAYER5_THICKNESS_VALUES= (0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002)
	LAYER5_DV= (0, 1, 0, 0, 0, 0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)

	# ---- STRINGER GEOMETRIC PARAMETERIZATION ----
	# STRINGERS= (2,GR-BMI)
	STRINGERS_BREAK_LOCATIONS= (0, 0.2, 0.4, 0.6, 0.8, 1)
	STRINGERS_ANGLES= (90, 270)
	# STRINGERS_HEIGHT_VALUES= EXTERIOR
	STRINGERS_THICKNESS_VALUES= (0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01)
	STRINGERS_DV= (0, 1, 2, 3, 4, 0, 0, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16)

	# ---- BAFFLE GEOMETRIC PARAMETERIZATION ----
	# BAFFLES= (5,PANEL)
	BAFFLES_LOCATION= (0, 0.2, 0.4, 0.6, 0.8)
	BAFFLES_THICKNESS= (0.01, 0.01, 0.01, 0.01, 0.01)
	# BAFFLES_HEIGHT= EXTERIOR
	BAFFLES_HALF_WIDTH= (1.1)
	BAFFLES_DV= (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
	  
	# ============================================================================
	# Build constraints & domain information
	# ============================================================================
	
	# -------------------------------- WALL --------------------------------------
	# Centerline constraints
	# For sampling purposes, we require overlap between the slope ranges on either side of the throat
	A1, b1 = bspline(WALL_COEFS1, WALL_COEFS1_DV, 0, (-0.3,0.3,-0.3,0.3), 
					 xLimits=[None,2.3], delta=0.2, minThroat=-0.05, maxThroat=0.1, output=output)
	# Major axis constraints
	A2, b2 = bspline(WALL_COEFS2, WALL_COEFS2_DV, 0, (0.05,1.2,[0.05,0.05,0.05,-0.2,-0.2,-0.1],[0.6,0.6,0.6,0.5,0.4,0.3]), 
					 xLimits=[None,2.3], delta=0.2, minThroat=0.4, output=output)
	# Minor axis constraints
	# Likewise for sampling here, we set max slope to 0.01, in reality this could be 0
	# For some reason the sampling does not deal well with 0 slopes
	A3, b3 = bspline(WALL_COEFS3, WALL_COEFS3_DV, 0, (-1.2,0.6,-0.3,0.6),
					 xLimits=[None,2.3], delta=0.2, minThroat=0.1, maxThroat=0.5, output=output)				 
	Awall, bwall = cleanupConstraintMatrix(Alist=[A1,A2,A3],blist=[b1,b2,b3])

	# Manually add coupling constraints for major/minor axis pre-throat slopes
	Acouple = np.zeros((2,18))
	bcouple = np.zeros((2,1))
	Acouple[0,10] = 1.
	Acouple[0,15] = 1.
	bcouple[0,0] = 0.439461*2
	Acouple[1,11] = 1.
	Acouple[1,10] = -1.
	Acouple[1,16] = 1.
	Acouple[1,15] = -1.
	Awall = np.vstack((Awall,Acouple))
	bwall = np.vstack((bwall,bcouple))

	inner_wall_domain = LinIneqDomain(Awall, np.squeeze(bwall), **kwargs)
	# As the range of each of these directions is less than one, 
	# this old code setup scaling for the normalized domain with lower and upper bounds 
	#x_wall, _ = inner_wall_domain.chebyshev_center() # use center shape as baseline
	#lb = x_wall - 0.5
	#ub = x_wall + 0.5

	# Instead we compute the extent in each direction 
	lb = np.zeros(len(inner_wall_domain))
	ub = np.zeros(len(inner_wall_domain))
	for i in range(len(inner_wall_domain)):
		ei = np.zeros(len(inner_wall_domain))
		ei[i] = 1
		lb[i] = inner_wall_domain.corner(-ei)[i]
		ub[i] = inner_wall_domain.corner(ei)[i]

	#inner_wall_domain = LinIneqDomain(Awall, np.squeeze(bwall), lb = lb, ub = ub, center = x_wall)
	inner_wall_domain = LinIneqDomain(Awall, np.squeeze(bwall), lb = lb, ub = ub, names = 'inner wall', **kwargs)
						 
	# -------------------------------- THERMAL LAYER -----------------------------
	A4, b4 = piecewiseBilinearAxial(LAYER1_THICKNESS_LOCATIONS, LAYER1_THICKNESS_ANGLES,
							   LAYER1_THICKNESS_VALUES, LAYER1_DV, (-0.2,0.2), 
							   xLimits=[0.,1.], deltax=0.15, deltay=60, deltaz=0.01,
							   output=output)						  
	Athermal, bthermal = cleanupConstraintMatrix(Alist=[A4],blist=[b4])
	x_thermal = list(LAYER1_THICKNESS_LOCATIONS) + list(LAYER1_THICKNESS_ANGLES) + \
		list(LAYER1_THICKNESS_VALUES);
	x_thermal = np.array([x_thermal[i] for i in range(len(x_thermal)) if LAYER1_DV[i] != 0 ]);
	lb = np.hstack((lb_perc*x_thermal[0], 0.01*np.ones(12)))
	ub = np.hstack((ub_perc*x_thermal[0], 0.05*np.ones(12)))
	#thermal_layer_domain = LinIneqDomain(Athermal, np.squeeze(bthermal), lb = lb, ub = ub, center = x_thermal)
	thermal_layer_domain = LinIneqDomain(Athermal, np.squeeze(bthermal), lb = lb, ub = ub, names = 'thermal layer', **kwargs)
	
	# -------------------------------- AIR GAP -----------------------------------
	air_gap_domain  = UniformDomain(0.003, 0.05, names = 'air gap')#, center = 0.0254)
	
	# -------------------------------- INNER LOAD LAYER --------------------------
	A5, b5 = piecewiseBilinearAxial(LAYER3_THICKNESS_LOCATIONS, LAYER3_THICKNESS_ANGLES,
							   LAYER3_THICKNESS_VALUES, LAYER3_DV, (-0.2,0.2), 
							   xLimits=[0.,1.], deltax=0.15, deltay=60, deltaz=0.01,
							   output=output) 
	Aload1, bload1 = cleanupConstraintMatrix(Alist=[A5],blist=[b5])  
	x_load1 = list(LAYER3_THICKNESS_LOCATIONS) + list(LAYER3_THICKNESS_ANGLES) + \
		list(LAYER3_THICKNESS_VALUES);					   
	x_load1 = np.array([x_load1[i] for i in range(len(x_load1)) if LAYER3_DV[i] != 0 ]);  
	lb = np.hstack((lb_perc*x_load1[0], 0.001*np.ones(12)))
	ub = np.hstack((ub_perc*x_load1[0], 0.006*np.ones(12)))
	load_layer_inner_domain = LinIneqDomain(Aload1, np.squeeze(bload1), lb = lb, ub = ub, names = 'inner load layer', **kwargs)#, center = x_load1)
	
	# -------------------------------- MIDDLE LOAD LAYER -------------------------
	A6, b6 = piecewiseBilinearAxial(LAYER4_THICKNESS_LOCATIONS, LAYER4_THICKNESS_ANGLES,
							   LAYER4_THICKNESS_VALUES, LAYER4_DV, (-0.2,0.2), 
							   xLimits=[0.,1.], deltax=0.15, deltay=60, deltaz=0.01,
							   output=output) 
	Aload2, bload2 = cleanupConstraintMatrix(Alist=[A6],blist=[b6])  
	x_load2 = list(LAYER4_THICKNESS_LOCATIONS) + list(LAYER4_THICKNESS_ANGLES) + \
		list(LAYER4_THICKNESS_VALUES);					   
	x_load2 = np.array([x_load2[i] for i in range(len(x_load2)) if LAYER4_DV[i] != 0 ]);  
	lb = np.hstack((lb_perc*x_load2[0], 0.0064*np.ones(12)))
	ub = np.hstack((ub_perc*x_load2[0], 0.0159*np.ones(12)))
	load_layer_middle_domain = LinIneqDomain(Aload2, np.squeeze(bload2), lb = lb, ub = ub, names = 'middle load layer', **kwargs)# , center = x_load2)
	
	# -------------------------------- OUTER LOAD LAYER --------------------------
	A7, b7 = piecewiseBilinearAxial(LAYER5_THICKNESS_LOCATIONS, LAYER5_THICKNESS_ANGLES,
							   LAYER5_THICKNESS_VALUES, LAYER5_DV, (-0.2,0.2), 
							   xLimits=[0.,1.], deltax=0.15, deltay=60, deltaz=0.01,
							   output=output) 
	Aload3, bload3 = cleanupConstraintMatrix(Alist=[A7],blist=[b7])  
	x_load3 = list(LAYER5_THICKNESS_LOCATIONS) + list(LAYER5_THICKNESS_ANGLES) + \
		list(LAYER5_THICKNESS_VALUES);					   
	x_load3 = np.array([x_load3[i] for i in range(len(x_load3)) if LAYER5_DV[i] != 0 ]);  
	lb = np.hstack((lb_perc*x_load3[0], 0.001*np.ones(12)))
	ub = np.hstack((ub_perc*x_load3[0], 0.006*np.ones(12)))
	load_layer_outer_domain = LinIneqDomain(Aload3, np.squeeze(bload3), lb = lb, ub = ub, names = 'outer load layer', **kwargs)#, center = x_load3)
	
	# -------------------------------- STRINGERS ---------------------------------
	A8, b8 = piecewiseBilinearAxial(STRINGERS_BREAK_LOCATIONS, STRINGERS_ANGLES,
									STRINGERS_THICKNESS_VALUES, STRINGERS_DV,
									(-0.2,0.2), xLimits=[0.,1.], deltax=0.1,
									deltay=None,deltaz=None,output=output);
	Astringers, bstringers = cleanupConstraintMatrix(Alist=[A8],blist=[b8])
	x_stringers = list(STRINGERS_BREAK_LOCATIONS) + list(STRINGERS_ANGLES) + \
				  list(STRINGERS_THICKNESS_VALUES);
	
	#x_stringers = np.array([x_stringers[i] for i in range(len(x_stringers)) if STRINGERS_DV[i] != 0]);
	# Using cheb center b/c provided "center" is on boundary 
	x_stringers = np.array([ 0.10329229,  0.59437729,  0.79062522,  0.89692417,  0.00468099,
		0.00428666,  0.00429567,  0.00561663,  0.00617409,  0.00661974,
		0.00655446,  0.00667583,  0.00563446,  0.00511958,  0.0062054 ,
		0.00682246])
	lb = np.hstack((x_stringers[0:4]-0.2, 0.002*np.ones(12)));
	ub = np.hstack((x_stringers[0:4]+0.2, 0.01*np.ones(12)));
	stringers_domain = LinIneqDomain(Astringers, np.squeeze(bstringers), lb = lb, ub = ub, names = 'stringers', **kwargs)#, center = x_stringers)
	
	# -------------------------------- BAFFLES -----------------------------------
	A9, b9 = baffles(BAFFLES_LOCATION, BAFFLES_THICKNESS, 
					 0., BAFFLES_DV, 0.1, 
					 0.26, output=output);
	Abaffles, bbaffles = cleanupConstraintMatrix(Alist=[A9],blist=[b9]);
	x_baffles = list(BAFFLES_LOCATION) + list(BAFFLES_THICKNESS);
	x_baffles = np.array([x_baffles[i] for i in range(len(x_baffles)) if BAFFLES_DV[i] != 0]);
	lb = np.hstack((x_baffles[0:4]-0.15, 0.0074*np.ones(5)));
	ub = np.hstack((x_baffles[0:4]+0.15, 0.0359*np.ones(5)));
	baffles_domain = LinIneqDomain(Abaffles, np.squeeze(bbaffles), lb = lb, ub = ub, names = 'baffles', **kwargs)#, center = x_baffles)

	
	# -------------------------------- FULL CONSTRAINTS --------------------------
	
	design_domain = TensorProductDomain([inner_wall_domain, thermal_layer_domain,
								 air_gap_domain, load_layer_inner_domain, 
								 load_layer_middle_domain, 
								 load_layer_outer_domain, stringers_domain,
								 baffles_domain])

	return design_domain


def buildRandomDomain(output='verbose', truncate = None):
	'''
	The standard random domain with 40 random variables.
	'''

	random_domains = [
	#			CMC_DENSITY, 1,
		LogNormalDomain(7.7803, 0.0182**2, truncate = truncate, names = ['CMC density (kg/m^3)']),
	#			CMC_ELASTIC_MODULUS, 1,
		LogNormalDomain(4.2047, 0.0551**2, scaling = 1e9, truncate = truncate, names = ['CMC elastic modulus (Pa)']),
	#			CMC_POISSON_RATIO, 1,
		UniformDomain(0.23, 0.43, names = ['CMC Poisson ratio']),
	#			 CMC_THERMAL_CONDUCTIVITY, 1,
		UniformDomain(1.37, 1.45, names = ['CMC thermal conductivity (W/m-K)']),
	#			CMC_THERMAL_EXPANSION_COEF, 1, 
		UniformDomain(0.228e-6, 0.252e-6, names = ['CMC thermal expansion coef. (1/K)']),	 
	#			CMC_PRINCIPLE_FAILURE_STRAIN, 1, 
		LogNormalDomain(-2.6694, 0.1421**2, scaling=1e-2, truncate = truncate, names = ['CMC principle failure strain'] ),
	#			CMC_MAX_SERVICE_TEMPERATURE, 1, 
		UniformDomain(963, 983, names = ['CMC max service temperature (K)']),
	#
	#
	#			GR-BMI_DENSITY, 1, 
		UniformDomain(1563, 1573, names = ['GR-BMI density (kg/m^3)']), 
	#			GR-BMI_ELASTIC_MODULUS, 2,
		UniformDomain(57e9, 63e9, names = ['GR-BMI elastic modulus 1 (Pa)']),
		UniformDomain(57e9, 63e9, names = ['GR-BMI elastic modulus 2 (Pa)']),
	#			GR-BMI_SHEAR_MODULUS, 1, 
		UniformDomain(22.6e9, 24.0e9, names = ['GR-BMI shear modulus (Pa)']),
	#			GR-BMI_POISSON_RATIO, 1,
		UniformDomain(0.334, 0.354, names = ['GR-BMI Poisson ratio']), 
	#			GR-BMI_MUTUAL_INFLUENCE_COEFS, 2, 
		UniformDomain(-0.1, 0.1, names = ['GR-BMI mutual influence coef 1']),
		UniformDomain(-0.1, 0.1, names = ['GR-BMI mutual influence coef 2']),
	#			GR-BMI_THERMAL_CONDUCTIVITY, 3,
		UniformDomain(3.208, 3.546, names = ['GR-BMI thermal conductivity 1 (W/m-K)']),
		UniformDomain(3.208, 3.546, names = ['GR-BMI thermal conductivity 2 (W/m-K)']),
		UniformDomain(3.243, 3.585, names = ['GR-BMI thermal conductivity 3 (W/m-K)']),
	#			GR-BMI_THERMAL_EXPANSION_COEF, 3,
		UniformDomain(1.16e-6, 1.24e-6, names = ['GR-BMI thermal expansion coef. 1 (1/K)']), 
		UniformDomain(1.16e-6, 1.24e-6, names = ['GR-BMI thermal expansion coef. 2 (1/K)']), 
		UniformDomain(-0.04e-6, 0.04e-6, names = ['GR-BMI thermal expansion coef. 3 (1/K)']),
	#			GR-BMI_LOCAL_FAILURE_STRAIN, 5,
		UniformDomain(0.675e-2, 0.825e-2, names = ['GR-BMI local failure strain 1']),#, center = 0.75e-2),
		UniformDomain(-0.572e-2, -0.494e-2, names = ['GR-BMI local failure strain 2']),#, center = -0.52e-2),
		UniformDomain(0.675e-2, 0.825e-2, names = ['GR-BMI local failure strain 3']),#, center = 0.75e-2),
		UniformDomain(-0.572e-2, -0.494e-2, names = ['GR-BMI local failure strain 4']),#, center = -0.52e-2),
		UniformDomain(0.153e-2, 0.187e-2, names = ['GR-BMI local failure strain 5']),#, center = 0.17e-2),
	#			GR-BMI_MAX_SERVICE_TEMPERATURE, 1,
		UniformDomain(500, 510, names = ['GR-BMI max service temperature (K)']),
	#
	#
	#			TI-HC_DENSITY, 1, 
		UniformDomain(177.77, 181.37, names = ['Ti-HC density (kg/m^3)']),
	#			TI-HC_ELASTIC_MODULUS, 1, 
		LogNormalDomain(0.6441, 0.0779**2, scaling = 1e9, truncate = truncate, names = ['Ti-HC elastic modulus (Pa)']),
	#			TI-HC_POISSON_RATIO, 1, 
		UniformDomain(0.160, 0.196, names = ['Ti-HC Poisson ratio']),
	#			TI-HC_THERMAL_CONDUCTIVITY, 1, 
		UniformDomain(0.680, 0.736, names = ['Ti-HC thermal conductivity (W/m-K)']),
	#			TI-HC_THERMAL_EXPANSION_COEF, 1, 
		UniformDomain(2.88e-6, 3.06e-6, names = ['Ti-HC thermal expansion coef. (1/K)']),
	#			TI-HC_YIELD_STRESS, 1,
		LogNormalDomain(2.5500, 0.1205**2, scaling = 1e6, truncate = truncate, names = ['Ti-HC yield stress (Pa)'] ),
	#			TI-HC_MAX_SERVICE_TEMPERATURE, 1, 
		UniformDomain(745, 765, names = ['Ti-HC max service temperature (K)']),
	#
	#
	#			AIR_THERMAL_CONDUCTIVITY, 1, 
		UniformDomain(0.0320, 0.0530, names = ['Air thermal conductivity (W/m-K)']),
	#			PANEL_YIELD_STRESS, 1, 
		LogNormalDomain(4.3191, 0.1196**2, scaling = 1e6, truncate = truncate, names = ['Panel yield stress (Pa)']), 
	#			INLET_PSTAG, 1, 
		LogNormalDomain(11.5010, 0.0579**2, truncate = truncate, names = ['Inlet stagnation pressure (Pa)'] ),
	#			INLET_TSTAG, 1, 
		LogNormalDomain(6.8615, 0.0119**2, truncate = truncate, names = ['Inlet stagnation temperature (K)'] ),
	#			ATM_PRES, 1, 
		LogNormalDomain(9.8386, 0.0323**2, truncate = truncate, names = ['Atmospheric pressure (Pa)'] ),
	#			ATM_TEMP, 1, 
		LogNormalDomain(5.3781, 0.0282**2, truncate = truncate, names = ['Atmospheric temperature (K)'] ),
	#			HEAT_XFER_COEF_TO_ENV, 1
		LogNormalDomain(2.5090, 0.2285, truncate = truncate, names = ['Heat transfer coef. to environment (W/m^2-K)']),
	]

	return TensorProductDomain(random_domains)
