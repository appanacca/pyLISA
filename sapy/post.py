# -*- coding: utf-8 -*-
"""
Created on Mon May 19 00:37:38 2014

@author: appanacca


"""

import numpy as np
import matplotlib.pyplot as plt
import sys as sys
import sapy.chebdif as cb
import scipy.linalg as lin
import scipy.interpolate as intp

import scipy.io

import sapy.blasius as bl
import numba as nb

import bokeh.plotting as bkpl
import bokeh.models as bkmd
from matplotlib.widgets import Button

import pdb as pdb
import matplotlib as mpl


class viz(object):
    """
    viz: perform some visualization on the data generated by the
    modal.fluid class
    """
    def __init__(self, in_data):
        # as input needs the in_data.npz with the simulation results
        data = np.load(in_data+".npz")
        self.file_name = in_data
        self.Re = data['Re']
        self.flow = data['flow']
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
        self.alpha = data['alpha']

    def plot_velocity(self):
        """plot the velocity profiles"""
        fig, ay = plt.subplots(figsize=(10, 10), dpi=100)
        lines = ay.plot(self.U, self.y, 'b+', self.dU, self.y, 'g',
                        self.ddU, self.y, 'r', self.aCD, self.y, 'm',
                        self.daCD, self.y, 'c', lw=2)

        mpl.rc('xtick', labelsize=40)
        mpl.rc('ytick', labelsize=40)

        ay.set_ylabel(r'$y$', fontsize=32)
        lgd = ay.legend((lines),
                        (r'$U$', r'$\partial U$',
                         r'$\partial^2 U$', r'$a^* C_D$',
                         r'$\partial a^* C_D$'),
                        loc=3, ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
        ay.set_ylim([0,5])
        # ax.set_xlim([np.min(time[2*T:3*T]),np.max(time[2*T:3*T])])
        ay.grid()
        # plt.tight_layout()
        # fig.savefig('RESULTS'+'couette.png', bbox_extra_artists=(lgd, ),
        #              bbox_inches='tight', dpi=50)
        plt.show()

    def plot_spectrum(self, postName='spectrum.txt', *args):
            mpl.rc('xtick', labelsize=20)
            mpl.rc('ytick', labelsize=20)
            """ plot the spectrum """
            self.eigv_re = np.real(self.eigv)
            self.eigv_im = np.imag(self.eigv)

            self.eigv_re_adj = np.real(self.eigv_adj)
            self.eigv_im_adj = np.imag(self.eigv_adj)

            #file_name = raw_input("Please enter spectrum file name: ")
            #np.savetxt(file_name, np.c_[self.eigv_re, self.eigv_im, self.eigv_re_adj, self.eigv_im_adj],
            #        fmt='%1.4e',  header='eigv_re    eigv_im    eigv_re_adj    eigv_im_adj')
            np.savetxt(self.file_name+postName, np.c_[self.eigv_re*self.alpha, self.eigv_im*self.alpha, self.eigv_re_adj*self.alpha, self.eigv_im_adj*self.alpha],
                            fmt='%1.4e',  header=str(self.option)+'\n'+'eigv_re    eigv_im    eigv_re_adj    eigv_im_adj')

            #  for i in np.arange(10):
            self.fig, ay = plt.subplots(figsize=(10, 6), dpi=100)
            plt.subplots_adjust(bottom=0.2)
            lines = ay.plot(self.eigv_re, self.eigv_im, 'bs', self.eigv_re_adj,
                    -self.eigv_im_adj, 'ro', markersize=10)
            ay.set_ylabel(r'$c_i$', fontsize=35)
            ay.set_xlabel(r'$c_r$', fontsize=35)
            # lgd = ay.legend((lines),(r'$U$',r'$\delta U$',r'$\delta^2 U$'),
            #                          loc=3, ncol=3, bbox_to_anchor=(0,1),
            #                          fontsize = 32)


            ay.set_ylim([-2, 1])
            ay.set_xlim([0.6, 1.2])
            ay.grid()
            # plt.tight_layout()
            #  fig.savefig('RESULTS'+'spectrum_couette.png',
            #           bbox_inches='tight', dpi=50)
            # plt.show(lines)

            cc = plt.axes([0.7, 0.05, 0.1, 0.075])
            dd = plt.axes([0.81, 0.05, 0.1, 0.075])

            b_next = Button(cc, 'Next eigv')
            b_next.on_clicked(self.new)

            b_close = Button(dd, 'Close')
            b_close.on_clicked(self.close)

            plt.show()

    def plot_eigf(self):
            mpl.rc('xtick', labelsize=15)
            mpl.rc('ytick', labelsize=15)


            sel_eig = self.fig.ginput(2)

            c_r_picked = (sel_eig[0][0] + sel_eig[1][0])/2
            c_i_picked = (sel_eig[0][1] + sel_eig[1][1])/2

            c_range = c_r_picked*(1+0j) + c_i_picked*(0+1j)
            n = np.argmin(np.abs(self.eigv - c_range))
            c_picked = self.eigv[n]

            adj_c_range = c_r_picked*(1+0j) - c_i_picked*(0+1j)
            adj_n = np.argmin(np.abs(self.eigv_adj - adj_c_range))
            adj_c_picked = self.eigv_adj[adj_n]

            self.eigfun_picked = self.eigf[:, n]
            print (c_picked)  # , lin.norm(self.eigfun_picked)
            print (adj_c_picked)
            print (n)

            if self.option['variables'] == 'v_eta':
                # needed in the case "Euler_wave" because only the half of the
                # point are in fact v the other part of the vector is alpha*v
                v = self.eigfun_picked[0:self.option['n_points']]
                u = np.dot((v/self.option['alpha']), self.D[0]) * (0+1j)

                fig2, (ay2, ay3) = plt.subplots(1, 2)  # , dpi=50)
                lines2 = ay2.plot(np.real(u), self.y, 'r',
                                  np.imag(u), self.y, 'g',
                                  np.sqrt(u*np.conjugate(u)), self.y, 'm', lw=2)
                ay2.set_ylabel(r'$y$', fontsize=32)
                ay2.set_xlabel(r"$u$", fontsize=32)

                lgd = ay2.legend((lines2),(r'$Re$',r'$Im$',r'$Mod$'),
                               loc=3, ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay2.set_ylim([0, 5])
                # ay2.set_xlim([-1, 1])
                ay2.grid()

                lines3 = ay3.plot(np.real(v), self.y, 'r', np.imag(v), self.y, 'g',
                                  np.sqrt(v*np.conjugate(v)), self.y, 'm', lw=2)
                ay3.set_ylabel(r'$y$', fontsize=32)
                ay3.set_xlabel(r"$v$", fontsize=32)


                # lgd = ay3.legend((lines3),(r'$Re$',r'$Im$',r'$Mod$'), loc=3,
                #                  ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay3.set_ylim([0, 5])
                # ay3.set_xlim([-1, 1])
                ay3.grid()

                fig3, ay4 = plt.subplots(1, 1)  # , dpi=50)
                lines4 = ay4.plot(np.real(self.eigf_adj[:, n]), self.y, 'r',
                        np.imag(self.eigf_adj[:, n]), self.y, 'g',
                        np.sqrt(u*np.conjugate(self.eigf_adj[:, n])), self.y, 'm', lw=2)


                ay4.set_ylabel(r'$y$', fontsize=32)
                ay4.set_xlabel(r"$v^\dagger$", fontsize=32)
                lgd = ay2.legend((lines2),(r'$Re$',r'$Im$',r'$Mod$'),
                               loc=3, ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay4.set_ylim([0, 5])
                # ay2.set_xlim([-1, 1])
                ay4.grid()

                plt.show()

            elif self.option['variables'] == 'p_u_v':
                p = self.eigfun_picked[0:self.N]
                u = self.eigfun_picked[self.N:2*self.N]
                v = self.eigfun_picked[2*self.N:3*self.N]

                p_re = np.real(p)
                p_im = np.imag(p)
                p_mod = np.real(np.sqrt(p*np.conjugate(p)))

                u_re = np.real(u)
                u_im = np.imag(u)
                u_mod = np.real(np.sqrt(u*np.conjugate(u)))

                v_re = np.real(v)
                v_im = np.imag(v)
                v_mod = np.real(np.sqrt(v*np.conjugate(v)))

                #file_name_fun = raw_input("Please enter function file name: ")
                #np.savetxt(file_name_fun+'.out', np.transpose([self.y, p_re, p_im, p_mod,
                #u_re, u_im, u_mod, v_re, v_im, v_mod]), fmt='%1.4e',
                #            header='y   p_re    p_im    p_mod   u_re    u_im    u_mod    v_re    v_im    v_mod')

                #np.savetxt(file_name_fun+'.out', np.transpose([self.y, p_re]), fmt='%1.4e',
                #                            header='y   p_re    p_im    p_mod   u_re    u_im    u_mod    v_re    v_im    v_mod')

                fig2, (ay1, ay2, ay3) = plt.subplots(1, 3)  # , dpi = 50)
                lines1 = ay1.plot(np.real(p), self.y, 'r', np.imag(p), self.y, 'g',
                                  np.sqrt(p*np.conjugate(p)), self.y, 'm', lw=2)


                ay1.set_ylabel(r'$y$', fontsize=32)
                ay1.set_xlabel(r"$p$", fontsize=32)
                lgd = ay1.legend((lines1), (r'$Re$', r'$Im$', r'$Mod$'), loc=3,
                                 ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay1.set_ylim([0, 5])
                # ay1.set_xlim([-1, 1])
                ay1.grid()

                lines2 = ay2.plot(np.real(u), self.y, 'r', np.imag(u), self.y, 'g',
                                  np.sqrt(u*np.conjugate(u)), self.y, 'm', lw=2)
                ay2.set_ylabel(r'$y$', fontsize=32)
                ay2.set_xlabel(r"$u$", fontsize=32)
                # lgd = ay2.legend((lines2), (r'$Re$', r'$Im$', r'$Mod$'), loc=3,
                #                  ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay2.set_ylim([0, 5])
                # ay2.set_xlim([-1, 1])
                ay2.grid()

                lines3 = ay3.plot(np.real(v), self.y, 'r', np.imag(v), self.y, 'g',
                                  np.sqrt(v*np.conjugate(v)), self.y, 'm', lw=2)


                ay3.set_ylabel(r'$y$', fontsize=32)
                ay3.set_xlabel(r"$v$", fontsize=32)
                # lgd = ay3.legend((lines3), (r'$Re$', r'$Im$', r'$Mod$'), loc=3,
                #                  ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay3.set_ylim([0, 5])
                # ay3.set_xlim([-1, 1])
                ay3.grid()

                # fig2.savefig('fun.png', bbox_inches='tight', dpi=150)

                p_adj = self.eigf_adj[0:self.N, n]
                u_adj = self.eigf_adj[self.N:2*self.N, n]
                v_adj = self.eigf_adj[2*self.N:3*self.N ,n]

                fig3, (ay4, ay5, ay6) = plt.subplots(1, 3)  # , dpi = 50)
                lines4 = ay4.plot(np.real(p_adj), self.y, 'r', np.imag(p_adj), self.y, 'g',
                                  np.sqrt(p_adj*np.conjugate(p_adj)), self.y, 'm', lw=2)


                ay4.set_ylabel(r'$y$', fontsize=32)
                ay4.set_xlabel(r"$p^\dagger$", fontsize=32)
                lgd = ay4.legend((lines4), (r'$Re$', r'$Im$', r'$Mod$'), loc=3,
                                 ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay4.set_ylim([0, 5])
                # ay1.set_xlim([-1, 1])
                ay4.grid()

                lines5 = ay5.plot(np.real(u_adj), self.y, 'r', np.imag(u_adj), self.y, 'g',
                                  np.sqrt(u_adj*np.conjugate(u_adj)), self.y, 'm', lw=2)


                ay5.set_ylabel(r'$y$', fontsize=32)
                ay5.set_xlabel(r"$u^\dagger$", fontsize=32)
                # lgd = ay2.legend((lines2), (r'$Re$', r'$Im$', r'$Mod$'), loc=3,
                #                  ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay5.set_ylim([0, 5])
                # ay2.set_xlim([-1, 1])
                ay5.grid()

                lines6 = ay6.plot(np.real(v_adj), self.y, 'r', np.imag(v_adj), self.y, 'g',
                                  np.sqrt(v_adj*np.conjugate(v_adj)), self.y, 'm', lw=2)


                ay6.set_ylabel(r'$y$', fontsize=32)
                ay6.set_xlabel(r"$v^\dagger$", fontsize=32)
                # lgd = ay3.legend((lines3), (r'$Re$', r'$Im$', r'$Mod$'), loc=3,
                #                  ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
                ay6.set_ylim([0, 5])
                # ay3.set_xlim([-1, 1])
                ay6.grid()

                p_re_adj = np.real(p_adj)
                p_im_adj = np.imag(p_adj)
                p_mod_adj = np.real(np.sqrt(p_adj*np.conjugate(p_adj)))

                u_re_adj = np.real(u_adj)
                u_im_adj = np.imag(u_adj)
                u_mod_adj = np.real(np.sqrt(u_adj*np.conjugate(u_adj)))

                v_re_adj = np.real(v_adj)
                v_im_adj = np.imag(v_adj)
                v_mod_adj = np.real(np.sqrt(v_adj*np.conjugate(v_adj)))


                #file_name_fun = raw_input("Please enter function file name: ")
                np.savetxt(self.file_name+"_fun"+'.txt', np.transpose([self.y, p_re, p_im, p_mod,
                u_re, u_im, u_mod, v_re, v_im, v_mod, p_re_adj, p_im_adj, p_mod_adj,
                u_re_adj, u_im_adj, u_mod_adj, v_re_adj, v_im_adj, v_mod_adj]), fmt='%1.4e',
                header=str(self.option)+'\n'+'y   p_re    p_im    p_mod   u_re    u_im    u_mod    v_re    v_im    v_mod \
                    p_re_adj    p_im_adj    p_mod_adj   u_re_adj    u_im_adj    u_mod_adj    v_re_adj    v_im_adj    v_mod_adj')

                plt.show()

    def new(self, event):
            self.plot_eigf()

    def close(self, event):
            plt.close()
