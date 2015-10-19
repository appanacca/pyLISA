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
import clencurt as cc_int

import scipy.linalg as lin
import scipy.interpolate as intp

import scipy.io

import blasius as bl
import numba as nb

import bokeh.plotting as bkpl
import bokeh.models as bkmd

import pdb as pdb

import numpy.random as rnd
import scipy.integrate as integ

import matplotlib as mpl
import sapy.modal as sa
import scipy.signal as sig


class sensitivity(object):
    """ these class compute the sensitivity of the spectrum
    due to changes in the velocity profile or the drag coefficient"""

    def __init__(self, max_norm, in_data, idx, per_U_heigth=5, per_cd_heigth=1, *args):
        # here above, eps is the maximum norm of the disturb

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
        self.alpha = self.option['alpha']

        self.idx = idx   # idx is the index of the eigenvalue
        # which is the one who want to compute the sensitivity
        # the idx should be find with the bokeh plot or the
        # plot spectrum function

        self.distribution_ky = np.linspace(0.001, 10, 1000)
        # rnd.choice() give a float between [-eps,+eps]
        # that will be the wave number of the sinusoidal perturbation
        # is it possible to change the associated probability

        self.max_norm = max_norm
        self.per_U_heigth = per_U_heigth
        self.per_cd_heigth = per_cd_heigth

        # pdb.set_trace()
        self.y_new = np.concatenate((np.linspace(0, 10, 10000),
                                     np.linspace(10.001, self.y[0],1000)))
        '''
        f_U = intp.interp1d(self.y, self.U)
        self.U = f_U(self.y_new)
        f_dU = intp.interp1d(self.y, self.dU)
        self.dU = f_dU(self.y_new)
        f_ddU = intp.interp1d(self.y, self.ddU)
        self.ddU = f_ddU(self.y_new)'''
        
        # initialize default perturbation of U and Cd as zeros
        self.delta_U = np.zeros(len(self.y_new))
        self.delta_cd = np.zeros(len(self.y_new))

        self.sim_param_values = data['sim_param_values']
        self.sim_param_keys = data['sim_param_keys']

    def u_pert_sin(self):
        # distribution = per_heigth * (10**x / 1e100)
        ky = rnd.choice(self.distribution_ky)
        self.delta_U = np.sin((2*np.pi)/ky * self.y[self.y <= self.per_U_heigth])
        norm = lin.norm(self.delta_U)
        eps = rnd.choice(self.distribution_eps)
        self.delta_U = eps/(norm) * self.delta_U
        n_tmp = self.N - len(self.delta_U)
        self.delta_U = np.concatenate((np.zeros(n_tmp), self.delta_U))
        # print ky, eps

        '''fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(self.delta_U, self.y, 'b', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        ay.grid()
        plt.show(lines)'''

    def u_pert(self, y0, eps):
        ky = np.pi / eps

        self.delta_U = np.zeros(len(self.y_new))
        self.delta_U[(self.y_new> y0-eps) & (self.y_new < y0+eps)] = (1 +
                np.cos(ky*(self.y_new[(self.y_new > y0-eps) & (self.y_new < y0+eps)]-y0)))

        norm_st = lin.norm(self.delta_U)

        a = self.max_norm / norm_st

        self.delta_U[(self.y_new > y0-eps) & (self.y_new < y0+eps)] = a*(1 +
                np.cos(ky*(self.y_new[(self.y_new > y0-eps) & (self.y_new < y0+eps)]-y0)))

        # print lin.norm(self.delta_U)

        fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(self.delta_U, self.y_new, 'b', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        ay.set_ylim([0,10])
        ay.grid()
        plt.show(lines)

    def cd_pert_sin(self):
        # distribution = per_heigth * (10**x / 1e100)
        ky = rnd.choice(self.distribution_ky)
        self.delta_cd = np.sin((2*np.pi)/ky * self.y[self.y <= self.per_cd_heigth])
        norm = lin.norm(self.delta_cd)
        eps = rnd.choice(self.distribution_eps)
        self.delta_cd = eps/(norm) * self.delta_cd
        n_tmp = self.N - len(self.delta_cd)
        self.delta_cd = np.concatenate((np.zeros(n_tmp), self.delta_cd))
        # print ky, eps

        '''fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(self.delta_cd, self.y, 'b', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        ay.grid()
        plt.show(lines)'''

    def cd_pert(self, y0, eps):
        ky = np.pi / eps

        self.delta_cd = np.zeros(len(self.y_new))
        self.delta_cd[(self.y_new > y0-eps) & (self.y_new < y0+eps)] = (1 +
                np.cos(ky*(self.y_new[(self.y_new > y0-eps) & (self.y_new < y0+eps)]-y0)))

        norm_st = lin.norm(self.delta_cd)

        a = self.max_norm / norm_st

        self.delta_cd[(self.y_new > y0-eps) & (self.y_new < y0+eps)] = a*(1 +
                np.cos(ky*(self.y_new[(self.y_new > y0-eps) & (self.y_new < y0+eps)]-y0)))

        # print lin.norm(self.delta_U)

        fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(self.delta_cd, self.y_new, 'b', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        ay.set_ylim([0,1.5])
        ay.grid()
        plt.show(lines)


    def c_per(self):
        i = (0 + 1j)
        if self.option['variables'] == 'v_eta':
            v = self.eigf[:, self.idx]
            v_adj = self.eigf_adj[:, self.idx]
        elif self.option['variables'] == 'p_u_v':
            v = self.eigf[2*self.N:3*self.N, self.idx]
            v_adj = self.eigf_adj[2*self.N:3*self.N, self.idx]

        v_adj_conj = np.conjugate(v_adj)

        f_norm = (v_adj_conj * np.dot((self.D[1] - self.alpha**2),v))
        normaliz = integ.trapz(f_norm[self.y<10], self.y[self.y<10])

        v_adj_conj = v_adj_conj / normaliz

        # TEST FOR NORMALIZATION
        # f_norm = (v_adj_conj * np.dot((self.D[1] - self.alpha**2),v))
        # normaliz = integ.trapz(f_norm, self.y)
        # print normaliz

        Gu = (v_adj_conj * np.dot((self.D[1] - self.alpha**2),v) -
              np.dot(self.D[1],v*v_adj_conj) -
              (i/self.alpha) *np.dot(self.D[0], v_adj_conj) * np.dot(self.D[0],  v) * self.aCD)
        dv_adj = np.gradient(v_adj_conj) / np.gradient(self.y)
        vv = v*v_adj_conj
        d_vv = np.gradient(vv) / np.gradient(self.y)
        dd_vv = np.gradient(d_vv) / np.gradient(self.y)
        
        Gu = (v_adj_conj * np.dot((self.D[1] - self.alpha**2),v) -
              dd_vv - (i/self.alpha) * dv_adj * np.dot(self.D[0],  v) * self.aCD)
        #Gu = np.dot(self.D[1],v*v_adj_conj)
        #pdb.set_trace()


        f_Gu = intp.interp1d(self.y, Gu)
        Gu = f_Gu(self.y_new)

        #Gu = sig.savgol_filter(np.dot(self.D[0], v_adj_conj), 9,8)
        #Gu = v*v_adj_conj#np.dot(self.D[0], v_adj_conj)
        # pdb.set_trace()
        fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(np.real(Gu), self.y_new, 'b', np.imag(Gu),
                        self.y_new, 'r', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        lgd = ay.legend((lines), (r'$Re$', r'$Im$'), loc=3,
                                 ncol=2, bbox_to_anchor=(0, 1), fontsize=32)
        ay.set_ylim([0,10])
        ay.grid()
        plt.show(lines)

        # pdb.set_trace()

        Gcd = -(i/self.alpha)*np.dot(self.D[0], v_adj_conj) * np.dot(self.D[0],
                v) * self.U * 0.552  # sarebbe a* da cambiare tutta
        # l'intefaccia per separare CD ed aCD

        f_Gcd = intp.interp1d(self.y, Gcd)
        Gcd = f_Gcd(self.y_new)

        '''fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(np.real(Gcd), self.y_new, 'b', np.imag(Gcd),
                        self.y_new, 'r', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        lgd = ay.legend((lines), (r'$Re$', r'$Im$'), loc=3,
                                 ncol=2, bbox_to_anchor=(0, 1), fontsize=32)
        ay.grid()
        ay.set_ylim([0,5])
        plt.show(lines)'''

        delta_c = (integ.trapz(Gu*self.delta_U, self.y_new) +
                       integ.trapz(Gcd*self.delta_cd, self.y_new))
        return delta_c

    def sens_spectrum(self, fig_name, per_variab='all', *args):
        eps = 0.1
        y0 = np.linspace(eps, 10-eps, 1000)
        it = np.arange(len(y0))
        #pdb.set_trace()
        delta_spectrum = np.zeros(len(y0), dtype=np.complex_)
        for i in it:
            if per_variab == 'u':
                #pdb.set_trace()
                self.u_pert(y0[i], eps)
            elif per_variab == 'cd':
                self.cd_pert(y0[i], eps)
            elif per_variab == 'all':
                self.u_pert(y0[i], eps)
                self.cd_pert(y0[i], eps)

            #pdb.set_trace()
            delta_spectrum[i] = self.c_per()
            print y0[i],'  ', delta_spectrum[i]

        re = np.real(delta_spectrum) + np.real(self.eigv[self.idx])
        im = np.imag(delta_spectrum) + np.imag(self.eigv[self.idx])

        fig, ay = plt.subplots(figsize=(20, 20), dpi=50)
        lines = ay.plot(re, im, 'ko', np.real(self.eigv[self.idx]),
            np.imag(self.eigv[self.idx]), 'r*', markersize=20)
        ay.set_ylabel(r'$c_i$', fontsize=32)
        ay.set_xlabel(r'$c_r$', fontsize=32)
        #ay.set_ylim([0.08081, 0.0812])
        #ay.set_xlim([0.91551, 0.9158])
        fig.savefig(fig_name, bbox_inches='tight', dpi=150)
        plt.show()

    def validation(self, pos, amp):
        """ check if the sensitivity of an eigenvalue is the same with the
        adjoint procedure, or with a simple superposition of the base flow
        plus the random perturbation:
            dc = c(U+dU) - c(U) = dc(adjoint) """

        self.u_pert(pos, amp)  # call the perturbation creator
        f_u = intp.interp1d(self.y_new, self.delta_U)
        self.delta_U = f_u(self.y)


        self.U = self.U + self.delta_U
        # after the u_pert() call the self.delta_U property is accessible

        d_delta_U =  np.gradient(self.delta_U) / np.gradient(self.y)
        dd_delta_U = np.gradient(d_delta_U) / np.gradient(self.y)

        self.dU = self.dU + d_delta_U
        self.ddU = self.ddU + dd_delta_U
        #self.ddU = sig.savgol_filter(self.ddU, 10, 2)


        # JUST A LITTLE VISUAL TEST TO SEE IF THE ADDITION OF
        # THE VELOCITY WORKS
        fig, ay = plt.subplots(figsize=(10, 10), dpi=50)
        lines = ay.plot(self.U, self.y, 'b', self.dU, self.y, 'g',
                        self.ddU, self.y, 'r', self.aCD, self.y, 'm',
                        self.daCD, self.y, 'c', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        lgd = ay.legend((lines),
                        (r'$U$', r'$\partial U$',
                         r'$\partial^2 U$', r'$a^* C_D$',
                         r'$\partial a^* C_D$'),
                        loc=3, ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
        ay.set_ylim([0,5])
        ay.grid()
        plt.show()

        dic = dict(zip(self.sim_param_keys, self.sim_param_values))
        f = sa.fluid(dic)

        # pdb.set_trace()

        f.diff_matrix()
        f.read_velocity_profile()
        f.mapping()        

        f.y = self.y
        f.U = self.U
        f.dU = self.dU
        f.ddU = self.ddU
        f.aCD = self.aCD
        f.daCD = self.daCD

        f.set_operator_variables()
        f.solve_eig()

        fig, ay = plt.subplots(figsize=(20, 20), dpi=50)
        lines = ay.plot(np.real(f.eigv),
            np.imag(f.eigv), 'r*', markersize=20)
        ay.set_ylabel(r'$c_i$', fontsize=32)
        ay.set_xlabel(r'$c_r$', fontsize=32)
        plt.show()

        # remove the infinite and nan eigenvectors, and their eigenfunctions
        selector = np.isfinite(f.eigv)
        f.eigv = f.eigv[selector]
        eigv_im = np.imag(f.eigv)
        idx_new = np.argmax(eigv_im)
        eigv_new = f.eigv[idx_new]
        
        print 'new:', eigv_new
        print 'old:', self.eigv[16]
        print 'diff:', eigv_new-self.eigv[16]

        self.u_pert(pos, amp)
        print self.c_per()
