from __future__ import division
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

data = np.load('norm_alpha_G_300_RE_157.npz')

alpha = data['alpha']
norm_guRe = data['norm_guRe']
norm_guIm = data['norm_guIm']
norm_gcdRe = data['norm_gcdRe']
norm_gcdIm = data['norm_gcdIm']


alpha = alpha[1:-1]
norm_guRe = norm_guRe[1:-1]
norm_guIm = norm_guIm[1:-1]
norm_gcdRe = norm_gcdRe[1:-1]
norm_gcdIm = norm_gcdIm[1:-1]

###### TRANSFORM THE SENSITIVITY TO delta_omega FROM delta_C #######

norm_guRe = norm_guRe*alpha
norm_guIm = norm_guIm*alpha
norm_gcdRe = norm_gcdRe*alpha
norm_gcdIm = norm_gcdIm*alpha

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
fig.savefig('norm_alpha_G.png', bbox_inches = 'tight',dpi = 200)
plt.show()

###### SAVE THE CURVES INTO A FILE ########
file_name = 'norm_sens_vs_alpha_G_RE_157.txt'
header = 'alpha  Re(Gu)    Im(Gu)    Re(Gcd)    Im(Gcd)'

np.savetxt(file_name ,np.transpose([alpha, norm_guRe, norm_guIm, norm_gcdRe, norm_gcdIm]), fmt='%.4e', delimiter=' ', newline='\n', header=header)
