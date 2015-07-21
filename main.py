from pyLisa import *

option={'flow':'DATA/G.txt', \
	'n_points':200, \
	'lc':0.16739, \
	'Ymax':1000, \
	'perturbation':{'alpha':0.6, \
			'Re':160}, \
	'variables':'v_eta', \
	'equation':'Euler_wave', \
	'mapping':['finite',[0,(46.7/13.8)]], \
	'plot_lim':[[-0.5,2],[-0.5,2]], \
	'Froude': 0.02,\
	'slope': 1.3e-5 }


f=fluid(option)

f.diff_matrix()
f.read_velocity_profile()
f.mapping()
f.interpolate()
#f.set_blasisus(f.y)

#f.infinite_mapping()
#f.set_hyptan()
#f.set_poiseuille()

f.choose_variables()

f.solve_eig()
f.plot_velocity()
f.plot_spectrum()

#f.omega_alpha_curves(0.0001,2,50)

#print f.y, f.U

"""
option={'flow':'hyp', \
	'n_points':200, \
	'lc':0.16739, \
	'Ymax':300, \
	'perturbation':{'alpha':0.1, \
			'Re':160}, \
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
#f.build_operator()
#f.BC2()
#f.solve_eig()
f.plot_velocity()

#f.plot_spectrum()
f.plot_LNS_eigspectrum()

#f.omega_alpha_curves(0.0001,0.18,10)
"""

"""
f.build_operator()
f.BC1()
f.solve_eig()
f.plot_spectrum()
"""


#f.superpose_spectrum(0.0001,2,50)

"""
a=np.linspace(0.0001,2,50)
omega_sel=np.zeros(len(a))
for i in np.arange(len(a)):
	f.set_perturbation(a[i],160)
	f.LNS()
	f.solve_eig()
	
	# MODE 2
	#print f.eigv[(f.eigv.real>0.83) & (f.eigv.real<0.844) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.02)]
	#omega=a[i]*f.eigv[(f.eigv.real>0.83) & (f.eigv.real<0.844) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.02)]

	# MODE 1
	#print f.eigv[(f.eigv.real>0.8) & (f.eigv.real<0.83) & (f.eigv.imag<0.013) & (f.eigv.imag>-0.035)]
	#omega=a[i]*f.eigv[(f.eigv.real>0.8) & (f.eigv.real<0.83) & (f.eigv.imag<0.013) & (f.eigv.imag>-0.035)]
	
	# MODE 4
	#print f.eigv[(f.eigv.real>0.854) & (f.eigv.real<0.873) & (f.eigv.imag<0.017) & (f.eigv.imag>-0.002)]
	#omega=a[i]*f.eigv[(f.eigv.real>0.854) & (f.eigv.real<0.873) & (f.eigv.imag<0.017) & (f.eigv.imag>-0.002)]
	
	# MODE 3
	if i==0:
		omega=np.array([0+0j])*a[i]
	elif i==1:
		omega=np.array([0+1.326e-2j])*a[i]
	elif i>1 and i<24:
		print i
		print f.eigv[(f.eigv.real>0.84) & (f.eigv.real<0.92) & (f.eigv.imag<0.1) & (f.eigv.imag>0.02)]
		omega=a[i]*f.eigv[(f.eigv.real>0.84) & (f.eigv.real<0.92) & (f.eigv.imag<0.1) & (f.eigv.imag>0.02)]

	else:
		print "last"
		print f.eigv[(f.eigv.real>0.872) & (f.eigv.real<0.8855) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.001)]
		omega=a[i]*f.eigv[(f.eigv.real>0.872) & (f.eigv.real<0.8855) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.001)]



	
	if len(omega)>1:
		omega_sel[i]=omega.imag[-1]
	else:
		omega_sel[i]=omega.imag
	print omega_sel[i], a[i]

	np.savez('mode_3',a,omega_sel)

	#cc.plot_LNS_eigspectrum()
	#cc.omega_alpha_curves(0.01,2,30)
fig, ay = plt.subplots(dpi=150)
ay.plot(a,omega_sel,lw=2)
ay.set_ylabel(r'$\omega_i$',fontsize=32)
ay.set_xlabel(r'$\alpha$',fontsize=32)
ay.set_ylim([-0.005,0.1])
ay.set_xlim([0, 2])
ay.grid()
fig.savefig('mode_3.png', bbox_inches='tight',dpi=150)     
plt.show()
"""