# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:37:38 2014

@author: appanacca


"""

import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import sapy.chebdif as cb
import sapy.clencurt as cc_int

import scipy.linalg as lin
import scipy.interpolate as intp

import scipy.io

import sapy.blasius as bl
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

    def __init__(self, in_data, idx, per_U_heigth=5, per_cd_heigth=1, show_f = True, *args):
        # here above, eps is the maximum norm of the disturb

        # as input needs the in_data.npz with the simulation results
        data = np.load(in_data+".npz")
        self.file_name = in_data
        self.Re = data['Re']
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

        self.a_ast = self.option['a_ast']
        self.integ_matrix = data['integ_matrix']

        self.d = self.option['d']
        self.h = self.option['h']
        self.y_itf = self.option['y_itf']
        self.K11 = self.option['K11']
        self.K22 = self.option['K22']

        self.idx = idx   # idx is the index of the eigenvalue
        # which is the one who want to compute the sensitivity
        # the idx should be find with the bokeh plot or the
        # plot spectrum function


        self.show_f = show_f

        self.distribution_ky = np.linspace(0.001, 10, 1000)
        # rnd.choice() give a float between [-eps,+eps]
        # that will be the wave number of the sinusoidal perturbation
        # is it possible to change the associated probability

        self.per_U_heigth = per_U_heigth
        self.per_cd_heigth = per_cd_heigth

        # initialize default perturbation of U and Cd as zeros
        self.perturb = np.zeros(len(self.y))

        self.sim_param_values = data['sim_param_values']
        self.sim_param_keys = data['sim_param_keys']

    def get_perturbation(self, y0, eps, gamma, shape='gauss', *args):
        if shape == 'cos': #`è in realtà un coseno ma alla "Gauss" è concentrato intorno ad un y0
            ky = np.pi / gamma

            self.perturb = np.zeros(len(self.y))
            self.perturb[(self.y> y0-gamma) & (self.y < y0+gamma)] = (1 +
                    np.cos(ky*(self.y[(self.y > y0-gamma) & (self.y <
                        y0+gamma)]-y0)))

            norm_st = lin.norm(self.perturb)
            a = eps / norm_st
            self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)] = a*(1 +
                        np.cos(ky*(self.y[(self.y > y0-gamma) & (self.y <
                            y0+gamma)]-y0)))

        elif shape == 'sin':
            ky = np.pi / gamma
            self.perturb = np.zeros(len(self.y))
            self.perturb[(self.y> y0-gamma) & (self.y < y0+gamma)] = np.sin(ky*(self.y[(self.y > y0-gamma) & (self.y <
                    y0+gamma)]-y0))
            norm_st = lin.norm(self.perturb)
            a = eps / norm_st

            self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)] = a*self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)]

        elif shape == 'p1':
            ky = gamma
            self.perturb = np.zeros(len(self.y))
            self.perturb[(self.y> y0-gamma) & (self.y < y0+gamma)] = ky*self.y[(self.y> y0-gamma) & (self.y < y0+gamma)]
            norm_st = lin.norm(self.perturb)
            a = eps / norm_st
            self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)] = a*self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)]

        elif shape == 'p2':
            ky = gamma
            self.perturb = np.zeros(len(self.y))
            self.perturb[(self.y> y0-gamma) & (self.y < y0+gamma)] = ky*(self.y[(self.y> y0-gamma) & (self.y < y0+gamma)])**2
            norm_st = lin.norm(self.perturb)
            a = eps / norm_st
            self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)] = a*self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)]

        elif shape == 'p3':
                ky = gamma
                self.perturb = np.zeros(len(self.y))
                self.perturb[(self.y> y0-gamma) & (self.y < y0+gamma)] = ky*(self.y[(self.y> y0-gamma) & (self.y < y0+gamma)])**3
                norm_st = lin.norm(self.perturb)
                a = eps / norm_st
                self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)] = a*self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)]

        elif shape == 'tanh':
            ky = gamma
            self.perturb = np.zeros(len(self.y))
            self.perturb[(self.y> y0-gamma) & (self.y < y0+gamma)] = np.tanh(self.y[(self.y> y0-gamma) & (self.y < y0+gamma)])
            norm_st = lin.norm(self.perturb)
            a = eps / norm_st
            self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)] = a*self.perturb[(self.y > y0-gamma) & (self.y < y0+gamma)]


        elif shape == 'gauss':
                self.perturb = np.zeros(len(self.y))
                self.perturb = np.exp((-(self.y - y0)**2)/gamma)

                norm_st = lin.norm(self.perturb)

                a = eps / norm_st
                self.perturb = a*np.exp((-(self.y - y0)**2)/gamma)

        mpl.rc('xtick', labelsize=25)
        mpl.rc('ytick', labelsize=25)

        """fig, ay = plt.subplots(figsize=(10, 10), dpi=100)
        lines = ay.plot(self.perturb, self.y, 'b*', lw=2)
        ay.set_ylabel(r'$y$', fontsize=32)
        ay.set_xlabel(r'$\delta U$', fontsize=32)
        ay.set_ylim([0,5])
        ay.grid()
        plt.show(lines)"""

    def c_per(self, obj='u', file_name='sens_fun.out', *args):
        i = (0 + 1j)
        if self.option['variables'] == 'v_eta':
            v = self.eigf[:, self.idx]
            v_adj = self.eigf_adj[:, self.idx]

            v_adj_conj = np.conjugate(v_adj)

            f_norm = (v_adj_conj * np.dot((self.D[1] - self.alpha**2),v))
            normaliz = np.sum(self.integ_matrix*f_norm)

            v_adj_conj = v_adj_conj / normaliz

            I = np.eye(len(v))

            # TEST FOR NORMALIZATION
            #f_norm = (v_adj_conj * np.dot((self.D[1] - self.alpha**2),v))
            #normaliz = np.sum(self.integ_matrix*f_norm)
            #print (normaliz)

            #self.Gu = (v_adj_conj * np.dot((self.D[1] - I*self.alpha**2),v) -
            #      np.dot(self.D[1],v*v_adj_conj) -
            #      (i/self.alpha) *np.dot(self.D[0], v_adj_conj) * np.dot(self.D[0],  v) * self.aCD)
            dv_adj = np.gradient(v_adj_conj) / np.gradient(self.y)
            vv = v*v_adj_conj
            d_vv = np.gradient(vv) / np.gradient(self.y)
            dd_vv = np.gradient(d_vv) / np.gradient(self.y)


            self.Gu = (v_adj_conj * np.dot((self.D[1] - I*self.alpha**2),v) -
                  dd_vv - (i/self.alpha) * dv_adj * np.dot(self.D[0],  v) * self.aCD)

            #self.Gu = (v_adj_conj * np.dot((self.D[1] - I*self.alpha**2),v)) -np.dot(self.D[1],v*v_adj_conj)

            self.Gcd = -(i/self.alpha)*np.dot(self.D[0], v_adj_conj) * np.dot(self.D[0],
                v) * self.U * self.a_ast  # sarebbe a* da cambiare tutta
                                     # l'intefaccia per separare CD ed aCD


        elif self.option['variables'] == 'p_u_v':
            print(self.idx)
            p = self.eigf[:,self.idx][0:self.N]
            u = self.eigf[:,self.idx][self.N:2*self.N]
            v = self.eigf[:,self.idx][2*self.N:3*self.N]

            p_adj = self.eigf_adj[:,self.idx][0:self.N]
            u_adj = self.eigf_adj[:,self.idx][self.N:2*self.N]
            v_adj = self.eigf_adj[:,self.idx][2*self.N:3*self.N]

            const_A = 1./np.max( np.sqrt(np.real(u)**2 +np.imag(u)**2))

            v = v*const_A
            u = u*const_A
            p = p*const_A

            v_adj_conj = np.conjugate(v_adj)
            u_adj_conj = np.conjugate(u_adj)


            f_norm = v_adj_conj*v + u*u_adj_conj
            #normaliz = np.sum(self.integ_matrix*f_norm)
            normaliz = -np.trapz(f_norm[100:], self.y[100:])

            const_B = (1./normaliz)

            I = np.eye(len(v))
            i = (0 +1j)

            v_adj_conj = v_adj_conj*const_B
            u_adj_conj = u_adj_conj*const_B

            p_adj = u_adj*np.conjugate(const_B)


            d_uv = np.gradient(v*u_adj_conj) / np.gradient(self.y)

            if self.option['equation'] == 'LNS_Darcy':

                self.Gu = ((i/self.alpha)*d_uv #np.dot(self.D[0], v*u_adj_conj)
                                    +v*v_adj_conj +u*u_adj_conj)#/normaliz

                self.theta = self.alpha*np.sqrt(self.K11/self.K22)*self.y_itf
                print("theta: ", self.theta)


                #self.beta = self.Re*(self.d/(self.a_ast*self.h))*np.sqrt(self.K11*self.K22)*self.alpha*np.tanh(self.theta)

                self.GK11 = -0.5*self.Re*i*(self.d/(self.a_ast*self.h))*p[-1]*np.conjugate(p_adj[-1])*(np.sqrt(self.K22/self.K11)*np.tanh(self.theta) +(self.alpha*self.y_itf)/(np.cosh(self.theta)**2))
                self.GK22 = -0.5*self.Re*i*(self.d/(self.a_ast*self.h))*(self.K11/self.K22)*p[-1]*np.conjugate(p_adj[-1])*(np.sqrt(self.K22/self.K11)*np.tanh(self.theta) -(self.alpha*self.y_itf)/(np.cosh(self.theta)**2))


            else:
                self.Gu = ((-i/self.alpha)*self.aCD*u*u_adj_conj +
                        (i/self.alpha)*d_uv #np.dot(self.D[0], v*u_adj_conj)
                        +v*v_adj_conj +u*u_adj_conj)#/normaliz

                self.Gcd = ((-(i*self.a_ast)/self.alpha)*self.U*u*u_adj_conj)#/normaliz

                np.savetxt(self.file_name+"_green"+".txt", np.c_[self.y, np.abs(self.Gu), np.abs(self.Gcd), np.real(self.Gu), np.imag(self.Gu), np.real(self.Gcd), np.imag(self.Gcd)],
                                            fmt='%1.4e',  header=str(self.option)+'\n'+'\n'+'MAX |self.Gu|:'+str(np.max(np.abs(self.Gu)))+'MAX |self.Gcd|:'+str(np.max(np.abs(self.Gcd)))+'\n'+'y   |self.Gu|    |self.Gcd|    self.Gu_real     self.Gu_imag     self.Gcd_real    self.Gcd_imag')


            f_norm = v_adj_conj*v + u*u_adj_conj
            normaliz = -np.trapz(f_norm[100:], self.y[100:])

            print("normaliz: ", normaliz)


            ######## ATTENTION;  HERE I TRANSFORM THE SENSITIVITY FROM delta_C to delta_OMEGA
            #self.Gu = self.Gu*self.alpha
            #self.Gcd = self.Gcd*self.alpha



        if(self.show_f == True):
            phase_Gu = np.arctan(np.imag(self.Gu)/np.real(self.Gu))

            mpl.rc('xtick', labelsize=15)
            mpl.rc('ytick', labelsize=15)

            fig, (ay1, ay2) = plt.subplots(1,2, figsize=(10, 10), dpi=100)
            lines = ay1.plot(np.real(self.Gu), self.y, 'r', np.imag(self.Gu),
                            self.y, 'g', np.abs(self.Gu), self.y, 'm', lw=2)
            #lines = ay1.plot(np.abs(self.Gu), self.y, 'm', lw=2)
            ay1.set_ylabel(r'$y$', fontsize=32)
            ay1.set_xlabel(r'$G_U$', fontsize=32)
            lgd = ay1.legend((lines), (r'$Re$', r'$Im$', r'$Mod$' ), loc=3,
                                     ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
            ay1.set_ylim([0,5])
            ay1.grid()

            lines = ay2.plot(np.real(self.Gcd), self.y, 'r', np.imag(self.Gcd),
                            self.y, 'g', np.abs(self.Gcd), self.y, 'm', lw=2)
            #lines = ay2.plot(np.abs(self.Gcd), self.y, 'm', lw=2)
            ay2.set_ylabel(r'$y$', fontsize=32)
            ay2.set_xlabel(r'$G_{CD}$', fontsize=32)
            ay2.grid()
            ay2.set_ylim([0,5])
            plt.show(lines)

        if obj == 'norm':
            if self.option['equation'] == 'LNS_Darcy':
                return lin.norm(np.real(self.Gu), ord=np.inf), lin.norm(np.imag(self.Gu), ord=np.inf), np.abs(np.real(self.GK11)), np.abs(np.imag(self.GK11)), np.abs(np.real(self.GK22)), np.abs(np.imag(self.GK22))
            else:
                return lin.norm(np.real(self.Gu), ord=np.inf), lin.norm(np.imag(self.Gu), ord=np.inf), lin.norm(np.real(self.Gcd), ord=np.inf), lin.norm(np.imag(self.Gcd), ord=np.inf)
        elif obj == 'u':
            delta_c = np.sum((self.Gu*self.perturb)*self.integ_matrix)  # +((+i/self.alpha)*v_adj_conj*d_p)*self.integ_matrix)
            return delta_c
        elif obj == 'cd':
            delta_c = np.sum((self.Gcd*self.perturb)*self.integ_matrix)
            return delta_c
        elif obj == 'all':
            delta_c = np.sum((self.Gu*self.perturb)*self.integ_matrix) + np.sum((self.Gcd*self.perturb)*self.integ_matrix)
            return delta_c

    def sens_spectrum(self, fig_name, amp, gamma , eig_idx,  shape='gauss',  obj='u',  *args):
        y0 = np.linspace(0, 2, 30)
        it = np.arange(len(y0))
        #pdb.set_trace()
        delta_spectrum = np.zeros(len(y0), dtype=np.complex_)
        delta_spectrum_stab = np.zeros(len(y0), dtype=np.complex_)

        for i in it:
            self.get_perturbation(y0[i], amp, gamma, shape)
            #pdb.set_trace()
            delta_spectrum[i] = self.c_per(obj)
            delta_spectrum_stab[i] = self.validation(y0[i], amp,  gamma, eig_idx, shape)

            print ('perturbation centre: ', y0[i])

        re = np.real(delta_spectrum) + np.real(self.eigv[self.idx])
        im = np.imag(delta_spectrum) + np.imag(self.eigv[self.idx])

        re_stab = np.real(delta_spectrum_stab)
        im_stab = np.imag(delta_spectrum_stab)


        fig, ay = plt.subplots(figsize=(20, 20), dpi=50)
        lines = ay.plot(re, im, 'ko', np.real(self.eigv[self.idx]),
            np.imag(self.eigv[self.idx]), 'r*', re_stab, im_stab, 'bo', markersize=20)
        ay.set_ylabel(r'$c_i$', fontsize=32)
        ay.set_xlabel(r'$c_r$', fontsize=32)
        #ay.set_ylim([0.08081, 0.0812])
        #ay.set_xlim([0.91551, 0.9158])
        ay.grid()
        fig.savefig(fig_name, bbox_inches='tight', dpi=150)
        plt.show()

    def validation(self, pos, amp, gamma, eig_idx, shape):
        """ check if the sensitivity of an eigenvalue is the same with the
        adjoint procedure, or with a simple superposition of the base flow
        plus the random perturbation:
            dc = c(U+dU) - c(U) = dc(adjoint) """

        self.get_perturbation(pos, amp, gamma, shape)  # call the perturbation creator
        self.U = self.U + self.perturb
        # after the u_pert() call the self.delta_U property is accessible
        d_delta_U = np.gradient(self.perturb) / np.gradient(self.y)
        dd_delta_U = np.gradient(d_delta_U) / np.gradient(self.y)
        self.dU = self.dU + d_delta_U
        self.ddU = self.ddU + dd_delta_U

        """self.cd_pert(pos, amp)
        self.aCD = self.aCD + self.delta_cd"""

        mpl.rc('xtick', labelsize=40)
        mpl.rc('ytick', labelsize=40)
        # JUST A LITTLE VISUAL TEST TO SEE IF THE ADDITION OF
        # THE VELOCITY WORKS
        """fig, ay = plt.subplots(figsize=(10, 10), dpi=100)
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
        plt.show()"""

        dic = dict(zip(self.sim_param_keys, self.sim_param_values))
        f = sa.fluid(dic)

        # pdb.set_trace()

        f.diff_matrix()
        f.integ_matrix()

        f.mapping()

        f.y = self.y
        f.U = self.U
        f.dU = self.dU
        f.ddU = self.ddU
        f.aCD = self.aCD
        f.daCD = self.daCD

        f.set_operator_variables()
        f.solve_eig()

        '''fig, ay = plt.subplots(figsize=(20, 20), dpi=50)
        lines = ay.plot(np.real(f.eigv),
            np.imag(f.eigv), 'r*', markersize=20)
        ay.set_ylabel(r'$c_i$', fontsize=32)
        ay.set_xlabel(r'$c_r$', fontsize=32)
        plt.show()'''

        # remove the infinite and nan eigenvectors, and their eigenfunctions
        selector = np.isfinite(f.eigv)
        f.eigv = f.eigv[selector]
        eigv_im = np.imag(f.eigv)
        idx_new = np.argmax(eigv_im)
        eigv_new = f.eigv[idx_new]

        print ('new:', eigv_new)
        print ('old:', self.eigv[eig_idx])
        print ('diff:', eigv_new-self.eigv[eig_idx])

        #self.get_perturbation(pos, amp, gamma)
        print ('adj:', self.c_per())
        print ('diff %', np.real((eigv_new-self.eigv[eig_idx]) -self.c_per())/np.real(self.c_per()) *100, np.imag((eigv_new-self.eigv[eig_idx]) -self.c_per())/np.imag(self.c_per()) *100)

        return( eigv_new)
