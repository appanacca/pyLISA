# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:37:38 2014

@author: appanacca


"""

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import chebdif as cb
import scipy.linalg as lin
import scipy.interpolate as intp

import scipy.io

import blasius as bl
import numba as nb

import bokeh.plotting as bkpl
import bokeh.models as bkmd

import pdb as pdb

import numpy.random as rnd


class sensitivity(object):
    """ these class compute the sensitivity of the spectrum
    due to changes in the velocity profile or the drag coefficient"""

    def __init__(self, eps, in_data, idx):
        # here above, eps is the maximum norm of the disturb
        self.eps = eps

        # as input needs the in_data.npz with the simulation results
        data = np.load(in_data)
        self.y = data['y']
        self.U = data['U']
        self.dU = data['dU']
        self.ddU = data['ddU']
        self.aCD = data['aCD']
        self.daCD = data['daCD']
        self.eigv = data['eigv']
        self.eigf = data['eigf']
        self.option = dict(zip(data['sim_param_keys'],
                           data['sim_param_values']))
        self.N = self.option['n_points']
        self.D = data['D']
        self.eigv_adj = data['adj_eigv']
        self.eigf_adj = data['adj_eigf']

        self.idx = idx   # idx is the index of the eigenvalue
        # which is the one who want to compute the sensitivity
        # the idx should be find with the bokeh plot or the
        # plot spectrum function

    def norm(self):
        # rnd.choice() give a float between [-eps,+eps]
        # is it possible to change the associated probability
        distribution = np.linspace(-self.eps/np.max(self.y),
                                   self.eps/np.max(self.y), 100)
        self.delta_U = rnd.choice(distribution) * np.ones(self.N)

        fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(self.delta_U, self.y, 'b', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        ay.grid()
        plt.show()
        
