from __future__ import division
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

data = np.load('norm_alpha_I_330.npz')

alpha = data['alpha'][:-2]
norm_guRe = data['norm_guRe'][:-2]
norm_guIm = data['norm_guIm'][:-2]
norm_gcdRe = data['norm_gcdRe'][:-2]*2 #legato al a* che cambia
norm_gcdIm = data['norm_gcdIm'][:-2]*2 #legato al a* che cambia

#####  HERE WE CHANGE THE POINTS IN THE GRAPH THAT ARE NOT CONVERGED ########

#300
norm_guRe[0] = 809.418064702
norm_guIm[0] =  973.964918185
norm_gcdRe[0] = 2*1354.18662976
norm_gcdIm[0] = 2*1688.50724624

#270
norm_guRe[11] = 3481.98841613
norm_guIm[11] = 3606.44466457
norm_gcdRe[11] = 2*182.877560579
norm_gcdIm[11] = 2*235.937637323

#270
norm_guRe[12] = 6791.15904223
norm_guIm[12] = 6461.77000668
norm_gcdRe[12] = 2*355.050938332
norm_gcdIm[12] = 2*426.79638206


######  PLOT THE RESULTS ######

fig, ay  =  plt.subplots(dpi = 200)
lines = ay.plot(alpha, norm_guRe, 'r', alpha, norm_gcdRe, 'b', alpha, norm_guIm, 'm', alpha, norm_gcdIm, 'c', lw = 2)
#ay.set_ylabel(r'$c_i$',fontsize = 32)
ay.set_xlabel(r'$\alpha$',fontsize = 32)
#ay.set_ylim([-0.02, 0.12])
#ay.set_xlim([0.8, 0.92])
lgd = ay.legend((lines), ( r'$R_e{G_U}$', r'$R_e{G_{C_D}}$', r'$I_m{G_U}$', r'$I_m{G_{C_D}}$'), loc=3,
                 ncol=1, bbox_to_anchor=(1, 0), fontsize=22)
ay.grid()
fig.savefig('norm_alpha.png', bbox_inches = 'tight',dpi = 200)
plt.show()
