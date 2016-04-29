
# coding: utf-8

# ## Validation

# In[1]:

from __future__ import division
import sapy.modal as sa
import sapy.post as po
import sapy.sensitivity as sn
import pdb as pdb


# In[2]:

option = {'flow': 'couette',
          'n_points': 80,
          'lc': 0.16739,
          'Ymax': 1000,
          'yi': 10,
          'alpha': 1.5,
          'Re': 500,
          'variables': 'p_u_v',
          'equation': 'LNS',
          'mapping': ['semi_infinite_PB', [0, (46.7/13.8)]],
          'Froude': 0.02,
          'slope': 1.3e-5}


# In[3]:

f = sa.fluid(option)
f.diff_matrix()
f.set_couette()
f.integ_matrix()

f.set_operator_variables()

f.solve_eig()
f.adjoint_spectrum('cont')
f.solve_eig_adj()

f.save_sim('cou_cont')


# In[4]:

v = po.viz('cou_cont.npz')
v.plot_velocity()
v.plot_spectrum()


# In[5]:
# 56 62 73
om = sn.sensitivity('cou_cont.npz', 73)
a, b, c, d = om.c_per(obj='norm')
print (a, b, c,d)

#om.sens_spectrum('ke_cd_N001_puv.png', 1e-7, 1e-4, 189, obj='u', shape='gauss') # eps, gamma
om.validation(1, 1e-7, 1e-4, 69, 'gauss')
