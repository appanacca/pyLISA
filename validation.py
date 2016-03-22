
# coding: utf-8

# ## Validation

# In[12]:

from __future__ import division
import sapy.modal as sa
import sapy.post as po
import sapy.sensitivity as sn
import pdb as pdb


# In[6]:

option = {'flow': 'couette',
          'n_points': 200,
          'lc': 0.16739,
          'Ymax': 1000,
          'yi': 10,
          'alpha': 1.5,
          'Re': 500,
          'variables': 'v_eta',
          'equation': 'LNS',
          'mapping': ['semi_infinite_PB', [0, (46.7/13.8)]],
          'Froude': 0.02,
          'slope': 1.3e-5}


# In[7]:

f = sa.fluid(option)
f.diff_matrix()
f.set_couette()
f.integ_matrix()

f.set_operator_variables()

f.solve_eig()
f.adjoint_spectrum_v_eta('disc')
f.solve_eig_adj()

f.save_sim('cou_disc')
f.check_adj()


# In[13]:

v = po.viz('cou_disc.npz')
v.plot_velocity()
v.plot_spectrum()


# In[14]:

om = sn.sensitivity(0.00001, 'cou_disc.npz', 378)
#om.u_pert(0.4, 0.2)
#om.cd_pert(0.5, 0.1)
#om.c_per()

#om.sens_spectrum('ke_u_N01_ve.png', per_variab='u')
om.validation(0, 0.01, 378)
