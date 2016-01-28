import sapy.modal as sa
import sapy.post as po
import sapy.sensitivity as sn
import pdb as pdb
import numpy as np

option = {'flow': 'DATA/G.txt',
          'n_points': 160,
          'lc': 0.16739,
          'Ymax': 1000,
          'yi': 5,
          'alpha': 0.4,
          'Re': 160,
          'variables': 'p_u_v',
          'equation': 'Euler_CD',
          'mapping': ['semi_infinite_PB', [0, (46.7/13.8)]],
          'Froude': 0.02,
          'slope': 1.3e-5}


f = sa.fluid(option)

f.diff_matrix()
f.integ_matrix()
f.read_velocity_profile()
f.mapping()

f.interpolate()

# f.set_blasisus(f.y)

# f.infinite_mapping()
# f.set_hyptan()
# f.set_poiseuille()

f.set_operator_variables()

f.solve_eig()
f.adjoint_spectrum_v_eta('cont')
f.solve_eig_adj()

f.save_sim('200_puv_cont')
#f.check_adj()



v = po.viz('200_puv_cont.npz')
v.plot_spectrum()
v.plot_velocity()
# f.omega_alpha_curves(0.0001,2,5

idx = np.argmax(np.imag(f.eigv))
om = sn.sensitivity('200_puv_cont.npz', idx)
om.c_per(obj='norm')

#om.sens_spectrum('ke_cd_N001_puv.png', 1e-3, 1e-2, obj='u', shape='sin') # eps, gamma
#om.validation(1, 1e-2, 1, 17, 'tanh')




"""
# PROCEDURE TO ANALYZE THE SINGLE MODES IN THE SPECTRUM
# it needs to be implemented in the "fluid" class and generalyzed in the
# interface

a = np.linspace(0.0001,2,50)
omega_sel = np.zeros(len(a))
for i in np.arange(len(a)):
     f.set_perturbation(a[i],160)
     f.LNS()
     f.solve_eig()

     # MODE 2
     #print f.eigv[(f.eigv.real>0.83) & (f.eigv.real<0.844) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.02)]
     #omega = a[i]*f.eigv[(f.eigv.real>0.83) & (f.eigv.real<0.844) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.02)]

     # MODE 1
     #print f.eigv[(f.eigv.real>0.8) & (f.eigv.real<0.83) & (f.eigv.imag<0.013) & (f.eigv.imag>-0.035)]
     #omega = a[i]*f.eigv[(f.eigv.real>0.8) & (f.eigv.real<0.83) & (f.eigv.imag<0.013) & (f.eigv.imag>-0.035)]

     # MODE 4
     #print f.eigv[(f.eigv.real>0.854) & (f.eigv.real<0.873) & (f.eigv.imag<0.017) & (f.eigv.imag>-0.002)]
     #omega = a[i]*f.eigv[(f.eigv.real>0.854) & (f.eigv.real<0.873) & (f.eigv.imag<0.017) & (f.eigv.imag>-0.002)]

     # MODE 3
     if i =  = 0:
          omega = np.array([0+0j])*a[i]
     elif i =  = 1:
          omega = np.array([0+1.326e-2j])*a[i]
     elif i>1 and i<24:
          print i
          print f.eigv[(f.eigv.real>0.84) & (f.eigv.real<0.92) & (f.eigv.imag<0.1) & (f.eigv.imag>0.02)]
          omega = a[i]*f.eigv[(f.eigv.real>0.84) & (f.eigv.real<0.92) & (f.eigv.imag<0.1) & (f.eigv.imag>0.02)]

     else:
          print "last"
          print f.eigv[(f.eigv.real>0.872) & (f.eigv.real<0.8855) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.001)]
          omega = a[i]*f.eigv[(f.eigv.real>0.872) & (f.eigv.real<0.8855) & (f.eigv.imag<0.02) & (f.eigv.imag>-0.001)]


     if len(omega)>1:
          omega_sel[i] = omega.imag[-1]
     else:
          omega_sel[i] = omega.imag
     print omega_sel[i], a[i]

     np.savez('mode_3',a,omega_sel)

     #cc.plot_LNS_eigspectrum()
     #cc.omega_alpha_curves(0.01,2,30)
fig, ay  =  plt.subplots(dpi = 150)
ay.plot(a,omega_sel,lw = 2)
ay.set_ylabel(r'$\omega_i$',fontsize = 32)
ay.set_xlabel(r'$\alpha$',fontsize = 32)
ay.set_ylim([-0.005,0.1])
ay.set_xlim([0, 2])
ay.grid()
fig.savefig('mode_3.png', bbox_inches = 'tight',dpi = 150)
plt.show()
"""
