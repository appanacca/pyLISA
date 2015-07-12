from pyLisa import *

option={'flow':'Hyperbolic_tangent', \
	'n_points':400, \
	'lc':0.16739, \
	'Ymax':300, \
	'perturbation':{'alpha':1.5, \
			'Re':1e20}, \
	'variables':'primitives', \
	'equation':'Euler', \
	'BC':'Neumann', \
	'plot_lim':[[-0.02,0.02],[0.83,0.85]]  }


f=fluid(option)

f.diff_matrix()
f.infinite_mapping()
f.set_hyptan()
f.plot_velocity()
f.LNS()
f.solve_eig()
print f.alpha
f.plot_LNS_eigspectrum()

#f.omega_alpha_curves(0.01,0.2,10)

"""
f.build_operator()
f.BC1()
f.solve_eig()
f.plot_spectrum()
"""
