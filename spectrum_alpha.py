from __future__ import division
import sapy.modal as sa
import sapy.post as po
import sapy.sensitivity as sn
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

"""
option = {'flow': 'DATA/G.txt',
          'n_points': 300,
          'lc': 0.16739,
          'Ymax': 1000,
          'yi': 10,
          'alpha': 0.6,
          'Re': 1e5,
          'variables': 'p_u_v',
          'equation': 'LNS_CD',
          'mapping': ['semi_infinite_PB', [0, (46.7/13.8)]],
          'Froude': 0.02,
          'slope': 1.3e-5}

f = sa.fluid(option)

f.diff_matrix()
f.integ_matrix()
f.read_velocity_profile()
f.mapping()
f.interpolate()

f.superpose_spectrum(0.1, 1, 10)

"""

a = np.linspace(0.1, 0.5, 9)
#c = np.arange(0.1, 1.1, 0.02)
#b = np.arange(1.1, 4.1, 0.1)
#a = np.concatenate((c,b))

eigv_sel = np.zeros(len(a))*(1 +0j)
norm_guRe = np.zeros(len(a))
norm_gcdRe = np.zeros(len(a))
norm_guIm = np.zeros(len(a))
norm_gcdIm = np.zeros(len(a))


for i in np.arange(len(a)):

    option = {'flow': 'DATA/H.txt',
              'a_ast': 1.104,
              'n_points': 300,
              'lc': 0.16739,
              'Ymax': 1000,
              'yi': 10,
              'alpha': a[i],
              'Re': 1e5,
              'variables': 'p_u_v',
              'equation': 'LNS_CD',
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

    idx = np.argmax(np.imag(f.eigv))
    #print idx
    eigv_sel[i] = f.eigv[idx]

    f.adjoint_spectrum('cont')
    f.solve_eig_adj()
    f.save_sim("temp")
    om = sn.sensitivity("temp", idx, show_f=False)
    norm_guRe[i], norm_guIm[i], norm_gcdRe[i], norm_gcdIm[i] = om.c_per(obj='norm')

file_name = option['flow'][-5]+"_"+str(option['Re'])

np.savez('norm_alpha_'+file_name, alpha=a, norm_guRe=norm_guRe, norm_guIm=norm_guIm, norm_gcdRe=norm_gcdRe, norm_gcdIm=norm_gcdIm)

#header = 'alpha  Gu_r  Gu_i  Gcd_r  Gcd_i'
#np.savetxt('norm_alpha'+name_file+'.txt' ,np.transpose([a, norm_guRe, norm_guIm, norm_gcdRe, norm_gcdIm]), fmt='%.4e', delimiter=' ', newline='\n', header=header)



fig, ay  =  plt.subplots(dpi = 100)
lines = ay.plot(a, norm_guRe, 'r', a, norm_gcdRe, 'b', a, norm_guIm, 'g', a, norm_gcdIm, 'c', lw = 2)
#ay.set_ylabel(r'$c_i$',fontsize = 32)
ay.set_xlabel(r'$\alpha$',fontsize = 32)
#ay.set_ylim([-0.02, 0.12])
#ay.set_xlim([0.8, 0.92])
lgd = ay.legend((lines), (r'$G_U$', r'$G_{C_D}$'), loc=3,
                 ncol=2, bbox_to_anchor=(0, 1), fontsize=32)
ay.grid()
fig.savefig('norm_alpha.png', bbox_inches = 'tight',dpi = 150)
plt.show()
