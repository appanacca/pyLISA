from __future__ import division
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

data = np.load('norm_alpha_G.npz')

alpha = data['alpha']
norm_guRe = data['norm_guRe']
norm_guIm = data['norm_guIm']
norm_gcdRe = data['norm_gcdRe']
norm_gcdIm = data['norm_gcdIm']

#####  HERE WE CHANGE THE POINTS IN THE GRAPH THAT ARE NOT CONVERGED ########


#320
norm_guRe[5] = 350.945481233
norm_guIm[5] = 331.118893343
norm_gcdRe[5] = 190.864465929
norm_gcdIm[5] = 226.89417846

#320
norm_guRe[17] = 466.136956889
norm_guIm[17] = 471.200201151
norm_gcdRe[17] = 30.804776564
norm_gcdIm[17] = 39.6721203132

#310
norm_guRe[18] = 744.867503753
norm_guIm[18] = 700.39684712
norm_gcdRe[18] = 37.5200355358
norm_gcdIm[18] = 52.2455284908

#320
norm_guRe[23] = 2439.13810107
norm_guIm[23] = 1962.71250848
norm_gcdRe[23] = 55.684784216
norm_gcdIm[23] = 77.6517252989


#320
norm_guRe[24] = 2336.51805653
norm_guIm[24] = 1991.36235666
norm_gcdRe[24] =  53.2016519402
norm_gcdIm[24] =  66.6075581996


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
