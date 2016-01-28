from __future__ import division
import sapy.modal as sa
import sapy.post as po
import sapy.sensitivity as sn
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

option = {'flow': 'DATA/G.txt',
          'n_points': 200,
          'lc': 0.16739,
          'Ymax': 1000,
          'yi': 10,
          'alpha': 0.6,
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

f.superpose_spectrum(0.1, 5, 49)


"""
#a = np.linspace(0.1,4, 10)
c = np.arange(0.1, 1.1, 0.02)
b = np.arange(1.1, 4.1, 0.1)
a = np.concatenate((c,b))

eigv_sel = np.zeros(len(a))*(1 +0j)
norm_gu = np.zeros(len(a))
norm_gcd = np.zeros(len(a))


for i in np.arange(len(a)):

    option = {'flow': 'DATA/G.txt',
              'n_points': 200,
              'lc': 0.16739,
              'Ymax': 1000,
              'yi': 10,
              'alpha': a[i],
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
    f.set_operator_variables()
    f.solve_eig()

    #idx = np.argmax(np.imag(f.eigv))
    #print idx
    #eigv_sel[i] = f.eigv[idx]

    #f.adjoint_spectrum_v_eta('cont')
    #f.solve_eig_adj()
    #f.save_sim('200_puv_cont')
    #om = sn.sensitivity('200_puv_cont.npz', idx)
    #norm_gu[i], norm_gcd[i] = om.c_per(obj='norm')
fig, ay  =  plt.subplots(dpi = 150)
lines = ay.plot(a, norm_gu, 'r', a, norm_gcd, 'b', lw = 2)
#ay.set_ylabel(r'$c_i$',fontsize = 32)
ay.set_xlabel(r'$\alpha$',fontsize = 32)
#ay.set_ylim([-0.02, 0.12])
#ay.set_xlim([0.8, 0.92])
lgd = ay.legend((lines), (r'$G_U$', r'$G_{C_D}$'), loc=3,
                 ncol=2, bbox_to_anchor=(0, 1), fontsize=32)
ay.grid()
fig.savefig('norm_alpha.png', bbox_inches = 'tight',dpi = 150)
plt.show()


fig, ay  =  plt.subplots(dpi = 150)
ay.plot(np.real(eigv), np.imag(eigv), 'ro', lw = 2)
ay.set_ylabel(r'$c_i$',fontsize = 32)
ay.set_xlabel(r'$c_r$',fontsize = 32)
ay.set_ylim([-0.02, 0.12])
ay.set_xlim([0.8, 0.92])
ay.grid()
fig.savefig('spectrum_alpha.png', bbox_inches = 'tight',dpi = 150)
plt.show()
"""
