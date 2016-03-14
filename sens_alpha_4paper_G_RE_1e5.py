from __future__ import division
import pdb as pdb
import numpy as np
import matplotlib.pyplot as plt

data = np.load('norm_alpha_G_300_Re_1e5.npz')

alpha = data['alpha']
norm_guRe = data['norm_guRe']
norm_guIm = data['norm_guIm']
norm_gcdRe = data['norm_gcdRe']
norm_gcdIm = data['norm_gcdIm']

#####  HERE WE CHANGE THE POINTS IN THE GRAPH THAT ARE NOT CONVERGED ########

#320
norm_guRe[5] = 23.9823723489
norm_guIm[5] = 27.4481022931
norm_gcdRe[5] = 13.5777107098
norm_gcdIm[5] = 18.0285824603

#320
norm_guRe[17] = 114.178494563
norm_guIm[17] = 99.8659103526
norm_gcdRe[17] = 7.34718332575
norm_gcdIm[17] = 9.02991645272

#310
norm_guRe[18] = 138.161828881
norm_guIm[18] = 122.678729019
norm_gcdRe[18] = 8.23371944525
norm_gcdIm[18] = 9.44076498559

#320
norm_guRe[23] = 494.983559849
norm_guIm[23] = 439.580916522
norm_gcdRe[23] = 14.7755873476
norm_gcdIm[23] = 13.9936420485

#320
norm_guRe[24] = 637.420024184
norm_guIm[24] = 540.9776788
norm_gcdRe[24] = 14.3230807882
norm_gcdIm[24] = 16.4516500374


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
file_name = 'norm_sens_vs_alpha_G.txt'
header = 'alpha  Re(Gu)    Im(Gu)    Re(Gcd)    Im(Gcd)'

np.savetxt(file_name ,np.transpose([alpha, norm_guRe, norm_guIm, norm_gcdRe, norm_gcdIm]), fmt='%.4e', delimiter=' ', newline='\n', header=header)
