
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


# In[3]:

g = sa.fluid(option)
g.diff_matrix()
g.integ_matrix()


# In[4]:

import numpy as np


# In[5]:

g.option['mapping'] = ['finite', [-5, 7]]


# In[6]:

g.mapping()


# In[6]


# In[7]:

g.u = np.sin(g.y)


# In[8]:

g.du = np.dot(g.D[0], g.u)


# In[9]:

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10,10), dpi=50)
lines = ax.plot(g.y,g.du,'r',g.y,g.u,'b',g.y,np.cos(g.y),'g',lw=2)
ax.set_ylabel(r'$c_i$',fontsize=32)
ax.set_xlabel(r'$c_r$',fontsize=32)
lgd=ax.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),loc = 3,ncol=3, bbox_to_anchor = (0,1),fontsize=32)
ax.grid()                                         
plt.tight_layout()
fig.savefig('figura.png', bbox_inches='tight',dpi=50, facecolor='none', edgecolor='none')     
plt.show(lines)


# In[10]:

b-a


# In[11]:

g.D[0]


# In[ ]:



