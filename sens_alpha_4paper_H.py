from __future__ import division
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

data = np.load('norm_alpha_H_300.npz')

alpha = data['alpha'][:-4]
norm_guRe = data['norm_guRe'][:-4]
norm_guIm = data['norm_guIm'][:-4]
norm_gcdRe = data['norm_gcdRe'][:-4]*2 #legato al a* che cambia
norm_gcdIm = data['norm_gcdIm'][:-4]*2 #legato al a* che cambia

#####  HERE WE CHANGE THE POINTS IN THE GRAPH THAT ARE NOT CONVERGED ########

#0.38571429  0.42142857  0.45714286  0.49285714 0.52857143


#310
norm_guRe[1] = 964.872087186
norm_guIm[1] = 1106.22931782
norm_gcdRe[1] = 2*839.820902934
norm_gcdIm[1] = 2*1141.45504793

#330
norm_guRe[2] = 491.376872804
norm_guIm[2] = 591.821891685
norm_gcdRe[2] = 2*289.276194367
norm_gcdIm[2] = 2*381.708413193

#310
norm_guRe[8] = 1745.73829385
norm_guIm[8] = 1968.16514214
norm_gcdRe[8] = 2*157.818980489
norm_gcdIm[8] = 2*217.306670296

#320
norm_guRe[9] = 3587.84047991
norm_guIm[9] = 3959.75778226
norm_gcdRe[9] = 2*260.763516553
norm_gcdIm[9] = 2*305.150056154

#270
norm_guRe[10] = 9252.64035839
norm_guIm[10] = 9086.65789456
norm_gcdRe[10] = 2*608.209938424
norm_gcdIm[10] = 2*731.824768482


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
