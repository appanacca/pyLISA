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
import scipy.sparse.linalg as lins
import scipy.interpolate as intp

import scipy.io

import blasius as bl
import numba as nb

import bokeh.plotting as bkpl
import bokeh.models as bkmd

import pdb as pdb
import scipy.integrate as integ


class fluid(object):
    """
    fluid: Perform a linear stability analysis after
    building the operator(ex.Orr-Sommerfeld)
    """
    def __init__(self, option, **kwargs):
        self.option = option
        self.N = option['n_points']
        # self.y = np.linspace(-1, 1, self.N)
        self.U = np.zeros(self.N)
        self.aCD = np.zeros(self.N)
        self.dU = np.zeros(self.N)
        self.ddU = np.zeros(self.N)
        self.alpha = option['alpha']
        self.Re = option['Re']

        self.Fr = option['Froude']
        self.slope = option['slope']

    @classmethod
    def init_provide_values(cls, in_data):
        # as input needs the in_data.npz with the simulation results
        data = np.load(in_data)
        cls.y = data['y']
        cls.U = data['U']
        cls.dU = data['dU']
        cls.ddU = data['ddU']
        cls.aCD = data['aCD']
        cls.daCD = data['daCD']
        cls.option = dict(zip(data['sim_param_keys'],
                          data['sim_param_values']))
        cls.N = data['sim_param_values'][-1]  # ['n_points']

        option = cls.y, cls.U, cls.dU, cls.ddU, cls.aCD, cls.daCD, cls.N
        return fluid(option)

    def diff_matrix(self):
        """build the differenziation matrix with chebichev discretization
        [algoritmh from Reddy & W...]"""
        # in this line we re-instanciate the y in gauss lobatto points
        self.y, self.D = cb.chebdif(self.N, 4)
        self.D = self.D + 0j
        # summing 0j is needed in order to make the D matrices immaginary

    def integ_matrix(self):
        """build the quadrature wheigth matrix with chebichev discretization,
        using Curtis- Clenshaw quadrature rules [algoritmh from Trefethen
        "Spectral methods in matlab"]"""
        self.integ_matrix = cc_int.clencurt(self.N-1)

    def read_velocity_profile(self):
        """ read from a file the velocity profile store in a
        .txt file and set the variable_data members"""
        in_txt = np.genfromtxt(self.option['flow'], delimiter=' ', skip_header=1)
        self.y_data = in_txt[:, 0]
        self.U_data = in_txt[:, 1]
        self.dU_data = in_txt[:, 2]
        self.ddU_data = in_txt[:, 3]
        self.aCD_data = in_txt[:, 4]
        self.daCD_data = in_txt[:, 5]
        self.lc = self.option['lc']  # 0.16739  #lc* = 0.22*(h-z1) / h

    def set_poiseuille(self):
        """set the members velocity and its derivatives as poiseuille flow"""
        Upoiseuille = (lambda y: 1-y**2)
        dUpoiseuille = (lambda y: -y*2)
        ddUpoiseuille = -np.ones(len(self.y))*2
        self.U = Upoiseuille(self.y)
        self.dU = dUpoiseuille(self.y)
        self.ddU = ddUpoiseuille
        self.aCD = np.zeros(self.N)
        self.daCD = np.zeros(self.N)

    def set_couette(self):
        """set the members velocity and its derivatives as couette flow"""
        Upoiseuille = (lambda y: y)
        dUpoiseuille = np.ones(len(self.y))
        ddUpoiseuille = np.zeros(len(self.y))
        self.U = Upoiseuille(self.y)
        self.dU = dUpoiseuille
        self.ddU = ddUpoiseuille
        self.aCD = np.zeros(self.N)
        self.daCD = np.zeros(self.N)


    def set_hyptan(self):
        """set the members velocity and its derivatives as hyperbolic tangent
        flow"""
        Uhyptan = (lambda y: 0.5*(1+np.tanh(y)))
        dUhyptan = (lambda y: 1/(2*np.cosh(y)**2))
        ddUhyptan = (lambda y: (1/np.cosh(y))*(-np.tanh(y)/np.cosh(y)))
        self.U = Uhyptan(self.y)
        self.dU = dUhyptan(self.y)
        self.ddU = ddUhyptan(self.y)
        self.aCD = np.zeros(self.N)
        self.daCD = np.zeros(self.N)

    def set_blasius(self, y_gl):
        """set the members velocity and its derivatives as boundary layer
        flow"""
        self.U, self.dU, self.ddU = bl.blasius(y_gl)
        self.aCD = np.zeros(len(self.y))
        self.daCD = np.zeros(self.N)

    def set_operator_variables(self):
        """ read the 'variable' option in the option
        dictionary and select the operator to solve"""

        if self.option['variables'] == 'v_eta':
            self.v_eta_operator()
        elif self.option['variables'] == 'p_u_v':
            self.LNS_operator()

    def mapping(self):
        if self.option['mapping'][0] == 'semi_infinite_PB':
            ymax = self.option['Ymax']
            s = self.y[1:-1]
            r = (s + 1)/2
            L = (ymax*np.sqrt(1-r[0]**2))/(2*r[0])

            # DERIVATIVE OF THE MAPPING FUNCTION, NEEDED FOR THE QUADRATURE
            # MATRIX
            map_integral = 8*L/(-(self.y + 1)**2 + 4)**(3/2)
            map_integral[0] = map_integral[1]*10

            self.y = (L*(s+1))/(np.sqrt((1 - ((s+1)**2)/4)))
            y_inf = 2*self.y[0]  # 2000
            self.y = np.concatenate([np.array([y_inf]), self.y])
            self.y = np.concatenate([self.y, np.array([0])])
            K = np.sqrt(self.y**2 + 4 * L**2)

            xi = np.zeros((self.N, 4))
            xi[:, 0] = 8 * L**2 / K**3
            xi[:, 1] = - 24 * self.y * L**2 / K**5
            xi[:, 2] = 96 * (self.y**2 - L**2) * L**2 / K**7
            xi[:, 3] = 480 * self.y * (3 * L**2 - self.y**2) * L**2 / K**9

            # MAP THE QUADRATURE WHEIGTH COEFFICIENT MATRIX
            self.integ_matrix = self.integ_matrix*map_integral

        elif self.option['mapping'][0] == 'semi_infinite_SH':
            ymax = self.option['Ymax']
            yi = self.option['yi']
            a = (yi*ymax)/(ymax - 2*yi)
            b = 1 + 2*a/ymax
            self.y = a*(1 + self.y)/(b - self.y)

            xi = np.zeros((self.N, 4))
            xi[:, 0] = a*(b + 1)/(b - self.y)**2
            xi[:, 1] = 2*a*(b + 1)/(b - self.y)**3
            xi[:, 2] = 6*a*(b + 1)/(b - self.y)**4
            xi[:, 3] = 24*a*(b + 1)/(b - self.y)**5

        elif self.option['mapping'][0] == 'semi_infinite_ZA':
            ymax = self.option['Ymax']
            self.y = (-ymax/2)*(-1 -self.y)

            xi = np.zeros((self.N, 4))
            xi[:, 0] = np.ones(len(self.y))*(2/ymax)
            xi[:, 1] = np.zeros(len(self.y))
            xi[:, 2] = np.zeros(len(self.y))
            xi[:, 3] = np.zeros(len(self.y))

        elif self.option['mapping'][0] == 'infinite':
            L = 10
            s_inf = 20
            s = (L/s_inf)**2
            self.y = (-L*self.y)/(np.sqrt(1+s-self.y**2))

            xi = np.zeros((self.N, 4))
            xi[:, 0] = (L**2 * np.sqrt(self.y**2 * (s + 1) /
                                       (L**2 + self.y**2)) /
                        (self.y*(L**2 + self.y**2)))
            xi[:, 1] = (-3*L**2*np.sqrt(self.y**2*(s + 1)/(L**2 + self.y**2)) /
                        (L**4 + 2*L**2*self.y**2 + self.y**4))
            xi[:, 2] = (3*L**2*np.sqrt(self.y**2*(s + 1)/(L**2 +
                        self.y**2))*(-L**2 + 4*self.y**2) /
                        (self.y*(L**6 + 3*L**4*self.y**2 +
                         3*L**2*self.y**4 + self.y**6)))
            xi[:, 3] = (L**2*np.sqrt(self.y**2*(s + 1)/(L**2 +
                        self.y**2))*(45*L**2 - 60*self.y**2) /
                        (L**8 + 4*L**6*self.y**2 + 6*L**4*self.y**4 +
                            4*L**2*self.y**6 + self.y**8))

        elif self.option['mapping'][0] == 'finite':
            a = self.option['mapping'][1][0]
            b = self.option['mapping'][1][1]
            self.y = (b - a) * 0.5 * self.y + (a + b) * 0.5

            xi = np.zeros((self.N, 4))
            xi[:, 0] = (2 * self.y - a - b) / (b - a)
            xi[:, 1] = np.zeros(self.N)
            xi[:, 2] = np.zeros(self.N)
            xi[:, 3] = np.zeros(self.N)

        self.D[0] = np.dot(np.diag(xi[:, 0]), self.D[0])
        self.D[1] = (np.dot(np.diag(xi[:, 0]**2), self.D[1]) +
                     np.dot(np.diag(xi[:, 1]), self.D[0]))
        self.D[2] = (np.dot(np.diag(xi[:, 0]**3), self.D[2]) +
                     3*np.dot(np.dot(np.diag(xi[:, 0]), np.diag(xi[:, 1])),
                              self.D[1]) +
                     np.dot(np.diag(xi[:, 2]), self.D[0]))
        self.D[3] = (np.dot(np.diag(xi[:, 0]**4), self.D[3]) +
                     6*np.dot(np.dot(np.diag(xi[:, 1]), np.diag(xi[:, 0]**2)),
                              self.D[2]) +
                     4 * np.dot(np.dot(np.diag(xi[:, 2]), np.diag(xi[:, 0])),
                                self.D[1]) +
                     3 * np.dot(np.diag(xi[:, 1]**2), self.D[1]) +
                     np.dot(np.diag(xi[:, 3]), self.D[0]))

        # scipy.io.savemat('test.mat', dict(x = self.D,y = xi)

    def v_eta_operator(self):
        """ this member build the stability operator in the variable v, so
        you have to eliminate pressure from the equation and get the
        u = f(v,alpha) from the continuity eq. """
        I = np.identity(self.N)
        i = (0+1j)
        delta = self.D[1] - self.alpha**2 * I
        Z = np.zeros((self.N, self.N))

        CD = np.matrix(np.diag(self.aCD))
        dCD = np.matrix(np.diag(self.daCD))
        U = np.matrix(np.diag(self.U))
        D1 = np.matrix(self.D[0])
        D2 = np.matrix(self.D[1])
        D4 = np.matrix(self.D[3])

        dU = np.matrix(np.diag(self.dU))
        ddU = np.matrix(np.diag(self.ddU))

        if self.option['equation'] == 'Euler':
            self.A = U * delta - np.diag(self.ddU)
            self.B = delta
        elif self.option['equation'] == 'Euler_CD':
            self.A = U * delta - np.diag(self.ddU)\
                    - (i/self.alpha) * (dCD*U*D1 + CD*dU*D1 + CD*U*D2)
            self.B = delta
        elif self.option['equation'] == 'Euler_CD_turb':
            print "not implemented yet"
        elif self.option['equation'] == 'LNS':
            self.A = (i/(self.alpha*self.Re)) *\
                    (D4 +I*self.alpha**4 -2*self.alpha**2 *D2)\
                    - ddU + U*delta
            self.B = delta
        elif self.option['equation'] == 'LNS_CD':
            self.A = (i/(self.alpha*self.Re)) *\
                    (delta)**2\
                    - ddU + U*delta - (i/self.alpha) *\
                    (dCD*U*D1 + CD*dU*D1 + CD*U*D2)
            self.B = delta
        elif self.option['equation'] == 'LNS_turb':
            print "not implemented yet"
        elif self.option['equation'] == 'LNS_turb_CD':
            print "not implemented yet"
        elif self.option['equation'] == 'Euler_wave':
            # in this case the B.C. is of 2nd order in omega so the matrix
            # problem should be reorganized see the article of Jerome
            # Hoepffner for details in the trick to transform polynomial
            # eigenvalue problem in a single one
            self.A = np.dot(np.diag(self.U), delta) - np.diag(self.ddU)
            self.B = delta
            self.C = Z

            A1 = np.concatenate((self.A, Z), axis=1)
            A2 = np.concatenate((Z, I), axis=1)
            self.A = np.concatenate((A1, A2))

            B1 = np.concatenate((self.B, self.C), axis=1)
            B2 = np.concatenate((I, Z), axis=1)
            self.B = np.concatenate((B1, B2))

        self.A_noBC = np.copy(self.A)
        self.B_noBC = np.copy(self.B)
        #self.A_noBC = self.A
        #self.B_noBC = self.B

        if self.option['equation'] == 'Euler':
            self.BC2()
        elif self.option['equation'] == 'Euler_CD':
            self.BC2()
        elif self.option['equation'] == 'Euler_CD_turb':
            print "not implemented yet"
        elif self.option['equation'] == 'LNS':
            self.BC1()
        elif self.option['equation'] == 'LNS_CD':
            self.BC1()
        elif self.option['equation'] == 'LNS_turb':
            print "not implemented yet"
        elif self.option['equation'] == 'LNS_turb_CD':
            print "not implemented yet"
        elif self.option['equation'] == 'Euler_wave':
            self.BC_wave_v_eta()

    def BC1(self):
        """impose the boundary condition as specified in the paper "Modal
        Stability Theory" ASME 2014 from Hanifi in his examples codes
           in v(0), v(inf) , Dv(0) , Dv(inf) all  = 0
        """

        eps = 1e-4*(0+1j)
        # v(inf) = 0
        self.A[0, :] = np.zeros(self.N)
        self.A[0, 0] = 1
        self.B[0, :] = self.A[0, :]*eps

        # v'(inf) = 0
        self.A[1, :] = self.D[0][0, :]
        self.B[1, :] = self.A[1, :]*eps

        # v'(0) = 0
        self.A[-2, :] = self.D[0][-1, :]
        self.B[-2, :] = self.A[-2, :]*eps

        # v(0) = 0
        self.A[-1, :] = np.zeros(self.N)
        self.A[-1, -1] = 1
        self.B[-1, :] = self.A[-1, :]*eps

    def BC_wave_v_eta(self):
        eps = 1e-4*(0+1j)

        # v(y_max)
        # self.A[0,:] = self.D[0][0,:]
        self.A[0, 0:self.N] = self.D[0][0, :]*self.U[0]**2\
            - (np.identity(self.N)[0, :])\
            * np.cos(self.slope)/self.Fr**2

        self.B[0, :] = np.concatenate((2 * self.U[0] * self.D[0][0, :],
                                       -self.D[0][0, :]), axis=1)
        # self.B[0,0] = +2*self.U[0]*self.D[0][0,0]
        # self.B[1,self.N] = -self.D[0][0,0]  #conditoon on C**2

        # v(0) = 0
        self.A[self.N - 1, :] = np.zeros(2*self.N)
        self.A[self.N - 1, self.N - 1] = 1
        self.B[self.N - 1, :] = self.A[self.N - 1, :]*eps

    def BC2(self):
        """impose the boundary condition as specified in the paper
        "Modal Stability Theory" ASME 2014 from Hanifi in his examples codes
           only in the v(0) and v(inf)  = 0
        """

        eps = 1e-4*(0+1j)

        # v(inf) = 0
        self.A[0, :] = np.zeros(self.N)
        self.A[0, 0] = 1
        self.B[0, :] = self.A[0, :]*eps

        # v(0) = 0
        self.A[-1, :] = np.zeros(self.N)
        self.A[-1, -1] = 1
        self.B[-1, :] = self.A[-1, :]*eps

    @nb.jit
    def solve_eig(self):
        """ solve the eigenvalues problem with the LINPACK subrutines"""
        self.eigv, self.eigf = lin.eig(self.A, self.B)
        # self.eigv, self.eigf = lins.eigs(self.A, k=10, M=self.B)
        # doesent work ARPACK problems with Arnoldi factorization

        # remove the infinite and nan eigenvectors, and their eigenfunctions
        selector = np.isfinite(self.eigv)
        self.eigv = self.eigv[selector]
        self.eigf = self.eigf[:, selector]

    def LNS_operator(self):
        # ----Matrix Construction-----------
        #  p |u |v
        # (       ) continuity
        # (       ) x-momentum
        # (       ) y-momentum

        I = np.identity(self.N)
        i = (0+1j)
        delta = self.D[1] - self.alpha**2 * I

        AA1 = np.zeros((self.N, self.N))
        AA2 = I
        AA3 = -(i/self.alpha)*self.D[0]

        AB1 = I

        if (self.option['equation'] == 'Euler_wave' or
                self.option['equation'] == 'Euler'):
                    AB2 = np.diag(self.U)
                    AC3 = np.diag(self.U)

        elif (self.option['equation'] == 'Euler_CD' or
              self.option['equation'] == 'Euler_CD_wave'):
                AB2 = np.diag(self.U) -(i/self.alpha)*np.diag(self.aCD*self.U)
                AC3 = + np.diag(self.U)

        elif self.option['equation'] == 'Euler_CD_turb':
            AB2 = (i*self.alpha*np.diag(self.U)-(2*self.lc**2) *
                   (np.dot(np.diag(self.dU),
                    self.D[1]) + np.dot(self.D[0], np.diag(self.ddU))))
            AC3 = + i*self.alpha*np.diag(self.U)

        elif self.option['equation'] == 'LNS':
            AB2 = np.diag(self.U) +(i/self.alpha)* delta/self.Re
            AC3 = np.diag(self.U) +(i/self.alpha)*delta/self.Re

        elif self.option['equation'] == 'LNS_CD':
            AB2 = (np.diag(self.U) +(i/self.alpha)* delta/self.Re
                        -(i/self.alpha)*np.diag(self.aCD*self.U))
            AC3 = + np.diag(self.U) +(i/self.alpha)* delta/self.Re

        elif self.option['equation'] == 'LNS_turb':
            AB2 = (i*self.alpha*np.diag(self.U) - delta/self.Re -
                   (2*self.lc**2) * (np.dot(np.diag(self.dU), self.D[1]) +
                                     np.dot(self.D[0], np.diag(self.ddU))))
            AC3 = + i*self.alpha*np.diag(self.U) - delta/self.Re

        elif self.option['equation'] == 'LNS_turb_CD':
            AB2 = (i*self.alpha*np.diag(self.U) - delta/self.Re +
                   np.diag(self.aCD*self.U) - (2*self.lc**2) *
                   (np.dot(np.diag(self.dU), self.D[1]) +
                       np.dot(self.D[0], np.diag(self.ddU))))
            AC3 = + i*self.alpha*np.diag(self.U) - delta/self.Re

        AB3 = -(i/self.alpha)*np.diag(self.dU)

        AC1 = -(i/self.alpha)*self.D[0]
        AC2 = np.zeros((self.N, self.N))

        BA1 = BA2 = BA3 = BB1 = BB3 = BC1 = BC2 = np.zeros((self.N, self.N))
        BB2 = BC3 =  I

        AA = np.concatenate((AA1, AA2, AA3), axis=1)
        AB = np.concatenate((AB1, AB2, AB3), axis=1)
        AC = np.concatenate((AC1, AC2, AC3), axis=1)

        self.A = np.concatenate((AA, AB, AC))

        BA = np.concatenate((BA1, BA2, BA3), axis=1)
        BB = np.concatenate((BB1, BB2, BB3), axis=1)
        BC = np.concatenate((BC1, BC2, BC3), axis=1)

        self.B = np.concatenate((BA, BB, BC))

        self.A_noBC = np.copy(self.A)
        self.B_noBC = np.copy(self.B)

        if self.option['equation'] == 'Euler':
            self.BC_LNS_neu_v()
        elif self.option['equation'] == 'Euler_wave':
            self.BC_LNS_wave()
        elif self.option['equation'] == 'Euler_CD':
            self.BC_LNS_neu_v()
        elif self.option['equation'] == 'Euler_CD_wave':
            self.BC_LNS_wave()
        elif self.option['equation'] == 'Euler_CD_turb':
            self.BC_LNS_neu_v()
        elif self.option['equation'] == 'LNS':
            self.BC_LNS_neu_u_v()
        elif self.option['equation'] == 'LNS_CD':
            self.BC_LNS_neu_u_v()
        elif self.option['equation'] == 'LNS_turb':
            self.BC_LNS_neu_u_v()
        elif self.option['equation'] == 'LNS_turb_CD':
            self.BC_LNS_neu_u_v()

    def BC_LNS_neu_u_v(self):
        idx_bc = np.array([self.N, 2*self.N, 2*self.N - 1, 3*self.N - 1])
        # index of the boundaries

        self.A[idx_bc, :] = np.zeros(3*self.N)
        self.B[idx_bc, :] = np.zeros(3*self.N)

        self.A[self.N, self.N] = 1
        self.A[2*self.N - 1, 2*self.N - 1] = 1

        self.A[2*self.N, 2*self.N] = 1
        self.A[3*self.N - 1, 3*self.N - 1] = 1

        # print self.A, self.B

    def BC_LNS_neu_v(self):
        idx_bc = np.array([2*self.N, 3*self.N - 1])
        self.A[idx_bc, :] = np.zeros(3*self.N)
        self.B[idx_bc, :] = np.zeros(3*self.N)

        self.A[2*self.N, 2*self.N] = 1
        self.A[3*self.N - 1, 3*self.N - 1] = 1

        #  print self.A, self.B

    def BC_LNS_wave(self):
        idx_bc = np.array([2*self.N, 3*self.N - 1])

        self.A[idx_bc, :] = np.zeros(3*self.N)
        self.B[idx_bc, :] = np.zeros(3*self.N)

        # v(0) = 0
        self.A[3*self.N - 1, 3*self.N - 1] = 1

        # v(y_max) --> equation
        self.A[2*self.N, 2*self.N] = - (np.cos(self.slope)/self.Fr**2)
        self.A[2*self.N, 0] = (0+1j)*self.alpha*self.U[0]
        self.B[2*self.N, 0] = (0+1j)*self.alpha

    def interpolate(self):
        f_U = intp.interp1d(self.y_data, self.U_data)
        idx = np.where(self.y < self.y_data[-1])
        y_int = self.y[idx]
        # pdb.set_trace()
        self.U = np.concatenate([(np.ones(len(self.y) -
                                  len(y_int))) * self.U_data[-1], f_U(y_int)])

        f_dU = intp.interp1d(self.y_data, self.dU_data)
        self.dU = np.concatenate([(np.ones(len(self.y) -
                                   len(y_int))) * 0, f_dU(y_int)])

        f_ddU = intp.interp1d(self.y_data, self.ddU_data)
        self.ddU = np.concatenate([(np.ones(len(self.y) -
                                    len(y_int))) * 0, f_ddU(y_int)])

        f_aCD = intp.interp1d(self.y_data, self.aCD_data)
        self.aCD = np.concatenate([(np.ones(len(self.y) - len(y_int))) * 0,
                                  f_aCD(y_int)])

        f_daCD = intp.interp1d(self.y_data, self.daCD_data)
        self.daCD = np.concatenate([(np.ones(len(self.y) - len(y_int))) * 0,
                                   f_daCD(y_int)])

    def adjoint_spectrum(self, method):
        I = np.identity(self.N)
        i = (0+1j)
        delta = self.D[1] - self.alpha**2 * I
        Z = np.zeros((self.N, self.N))

        CD = np.matrix(np.diag(self.aCD))
        dCD = np.matrix(np.diag(self.daCD))
        U = np.matrix(np.diag(self.U))
        D1 = np.matrix(self.D[0])
        D2 = np.matrix(self.D[1])
        D4 = np.matrix(self.D[3])

        dU = np.matrix(np.diag(self.dU))
        ddU = np.matrix(np.diag(self.ddU))

        if method == 'cont':
            if self.option['variables'] == 'v_eta':
                self.C = (2 * dU * D1 + U*delta +
                        ((i/self.alpha)*I)*D1*(CD*U*D1))
                        #((i/self.alpha)*I)*(dCD*U*D1 + CD*dU*D1 + CD*U*D2))
                self.E = delta

                """self.C = (-i/(self.alpha*self.Re)) *\
                         (D4 +I*self.alpha**4 -2*self.alpha**2 *D2)\
                        + 2*dU*D1 + U*delta
                self.E = delta"""
            elif self.option['variables'] == 'p_u_v':
                # ----Matrix Construction-----------
                #  p |u |v
                # (       ) continuity
                # (       ) x-momentum
                # (       ) y-momentum

                I = np.identity(self.N)
                i = (0+1j)
                delta = self.D[1] - self.alpha**2 * I

                AA1 = np.zeros((self.N, self.N))
                AA2 = I
                AA3 = (i/self.alpha)*D1

                AB1 = I
                AB2 = U - (i/self.alpha)*U*CD +(i/self.alpha)*(delta/self.Re)
                AB3 = np.zeros((self.N, self.N))

                AC1 = (i/self.alpha)*D1
                AC2 = -(i/self.alpha)*dU
                AC3 = U +(i/self.alpha)*(delta/self.Re)

                BA1 = BA2 = BA3 = BB1 = BB3 = BC1 = BC2 = np.zeros((self.N, self.N))
                BB2 = BC3 = I

                AA = np.concatenate((AA1, AA2, AA3), axis=1)
                AB = np.concatenate((AB1, AB2, AB3), axis=1)
                AC = np.concatenate((AC1, AC2, AC3), axis=1)

                self.C = np.conjugate(np.concatenate((AA, AB, AC)))

                BA = np.concatenate((BA1, BA2, BA3), axis=1)
                BB = np.concatenate((BB1, BB2, BB3), axis=1)
                BC = np.concatenate((BC1, BC2, BC3), axis=1)

                self.E = np.concatenate((BA, BB, BC))


        elif method == 'disc':
            self.C = np.conjugate(np.transpose(self.A_noBC))
            self.E = np.conjugate(np.transpose(self.B_noBC))
            if self.option['variables'] == 'p_u_v':
                 self.M = (np.diag(np.concatenate((self.integ_matrix,
                     self.integ_matrix, self.integ_matrix))))
            elif self.option['variables'] == 'v_eta':
                 self.M = np.diag(self.integ_matrix)

            self.C = np.matrix(self.C)
            self.E = np.matrix(self.E)
            self.M = np.matrix(self.M)
            M_inv = lin.inv(self.M)
            M_t = np.transpose(self.M)

            self.C = (M_inv*self.C)*M_t
            self.E = (M_inv*self.E)*M_t

        """impose the boundary condition as specified in the paper
        "Modal Stability Theory" ASME 2014 from Hanifi in his examples codes
           only in the v(0) and v(inf)  = 0
        """
        if self.option['variables'] == 'v_eta':
            eps = 1e-4*(0+1j)

            # v(inf) = 0
            self.C[0, :] = np.zeros(self.N)
            self.C[0, 0] = 1
            self.E[0, :] = self.C[0, :]*eps

            # v(0) = 0
            self.C[-1, :] = np.zeros(self.N)
            self.C[-1, -1] = 1
            self.E[-1, :] = self.C[-1, :]*eps

            # v'(inf) = 0
            self.C[1, :] = self.D[0][0, :]
            self.E[1, :] = self.C[1, :]*eps

            # v'(0) = 0
            self.C[-2, :] = self.D[0][-1, :]
            self.E[-2, :] = self.C[-2, :]*eps

        elif self.option['variables'] == 'p_u_v':

            if  (self.option['equation'] == 'Euler_CD' or
                  self.option['equation'] == 'Euler'):
                idx_bc = np.array([2*self.N, 3*self.N - 1])
                self.C[idx_bc, :] = np.zeros(3*self.N)
                self.E[idx_bc, :] = np.zeros(3*self.N)

                self.C[2*self.N, 2*self.N] = 1
                self.C[3*self.N - 1, 3*self.N - 1] = 1

            elif  (self.option['equation'] == 'LNS' or
                    self.option['equation'] == 'LNS_CD'):
                idx_bc = np.array([self.N, 2*self.N, 2*self.N - 1, 3*self.N - 1])
                # index of the boundaries

                self.C[idx_bc, :] = np.zeros(3*self.N)
                self.E[idx_bc, :] = np.zeros(3*self.N)

                self.C[self.N, self.N] = 1
                self.C[2*self.N - 1, 2*self.N - 1] = 1

                self.C[2*self.N, 2*self.N] = 1
                self.C[3*self.N - 1, 3*self.N - 1] = 1


    @nb.jit
    def solve_eig_adj(self):
        """ solve the eigenvalues problem with the LINPACK subrutines"""
        self.eigv_adj, self.eigf_adj = lin.eig(self.C, self.E)

        # self.eigv_adj, self.eigf_adj = lins.eigs(self.C, k=10, M=self.E)
        # doesent work ARPACK problems with Arnoldi factorization

        # remove the infinite and nan eigenvectors, and their eigenfunctions
        selector = np.isfinite(self.eigv_adj)
        self.eigv_adj = self.eigv_adj[selector]
        self.eigf_adj = self.eigf_adj[:, selector]





    @nb.jit
    def omega_alpha_curves(self, alpha_start, alpha_end, n_step, name_file='omega_alpha'):
        self.vec_alpha = np.linspace(alpha_start, alpha_end, n_step)
        self.vec_eigv_im = np.zeros(n_step)
        self.vec_eigv_re = np.zeros(n_step)

        for i in np.arange(n_step):
            self.set_perturbation(self.vec_alpha[i], self.Re)
            self.set_operator_variables()
            self.solve_eig()
            # self.vec_eigv_im[i] = np.max(self.eigv_im)

            #self.eigv = self.eigv[ (np.real(self.eigv)>0.92) &  (np.real(self.eigv)<1.1)]

            self.vec_eigv_im[i] = (self.vec_alpha[i] * np.max(np.imag(self.eigv)))
            self.vec_eigv_re[i] = (self.vec_alpha[i] * np.real(self.eigv[np.argmax(np.imag(self.eigv))]) )

            # print self.eigv_im
        np.savez('omega_alpha_'+name_file, self.vec_alpha, self.vec_eigv_im, self.vec_eigv_re)

        header = 'alpha  omega_i  omega_r'
        np.savetxt('omega_alpha_'+name_file+'.txt' ,np.transpose([self.vec_alpha, self.vec_eigv_im, self.vec_eigv_im]), fmt='%.4e', delimiter=' ', newline='\n', header=header)


        fig = plt.figure(dpi=150)
        ay1 = fig.add_subplot(111)
        ay1.plot(self.vec_alpha, self.vec_eigv_im, 'b', lw=2)
        ay1.set_ylabel(r'$\omega_i$', fontsize=32, color='b')
        ay1.set_xlabel(r'$\alpha$', fontsize=32)
        # lgd = ay.legend((lines), (r'$U$', r'$\delta U$', r'$\delta^2 U$'),
        #                 loc=3, ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
        # ay.set_ylim([-1, 0.1])
        # ay.set_xlim([0, 1.8])
        ay1.grid()
        for tl in ay1.get_yticklabels():
            tl.set_color('b')

        ay2 = ay1.twinx()
        ay2.plot(self.vec_alpha, self.vec_eigv_re, 'r', lw=2)
        ay2.set_ylabel(r'$\omega_r$', fontsize=32, color='r')
        ay2.grid()
        for tl in ay2.get_yticklabels():
                tl.set_color('r')

        # plt.tight_layout()
        fig.savefig('omega_alpha_'+name_file+'.png', bbox_inches='tight', dpi=150)
        plt.show()



    def set_perturbation(self, a, Re):
        self.alpha = a
        self.Re = Re

    def superpose_spectrum(self, alpha_start, alpha_end, n_step):
        self.vec_alpha = np.linspace(alpha_start, alpha_end, n_step)

        sp_re = np.array([])
        sp_im = np.array([])



        COLORS = ['aqua', 'blue', 'fuchsia', 'gold', 'green', 'orange', 'red',
                  'sienna', 'yellow', 'lime']   # from css list

        sp_re = []
        sp_im = []
        alpha = []
        col = []

        jj = -1
        for i in np.arange(n_step):
            jj = jj+1
            self.set_perturbation(self.vec_alpha[i], self.Re)
            self.set_operator_variables()
            self.solve_eig()
            self.eigv_re = np.real(self.eigv)
            self.eigv_im = np.imag(self.eigv)
            sp_re = sp_re + self.eigv_re.tolist()
            sp_im = sp_im + self.eigv_im.tolist()
            alpha = alpha + (np.ones(len(self.eigv_im))*self.vec_alpha[i]).tolist()
            if jj > 9:
                jj = jj-10    # these with the above inizialization is needed for
            # the iteration in the COLOURS list
            col = col + [COLORS[jj]]*len(self.eigv_im)

        source = bkpl.ColumnDataSource(
                data=dict(
                    x = sp_re,
                    y = sp_im,
                    a = alpha,
                    color = col,
                ))

        hover = bkmd.HoverTool(
        tooltips=[
            ("index", "$index"),
            ("(Re, Im)", "(@x, @y)"),
            ("alpha", "@a"),
        ]
        )

        bkpl.output_file("spectrum.html")

        TOOLS = "resize, crosshair, pan, wheel_zoom, box_zoom, reset,\
                            box_select, lasso_select"

        p = bkpl.figure(plot_width=1000, plot_height=600, tools=[hover, TOOLS],
                                title="Superimposed Spectrum for varing wavenumbers", x_axis_label='c_r',
                                y_axis_label='c_i', x_range=(0.8, 0.96),
                                y_range=(-0.05, 0.1))

        p.circle('x', 'y', size=12.5, source=source, fill_color='color')

        bkpl.show(p)

    def save_sim(self, file_name):
        np.savez(file_name, sim_param_keys=np.array(self.option.keys()),
                 sim_param_values=np.array(self.option.values(), dtype=object),
                 U=self.U, dU=self.dU, y=self.y,
                 ddU=self.ddU, aCD=self.aCD, daCD=self.daCD,
                 eigv=self.eigv, eigf=self.eigf, D=self.D,
                 adj_eigv=self.eigv_adj, adj_eigf=self.eigf_adj,
                 integ_matrix=self.integ_matrix, alpha=self.alpha, Re=self.Re, flow=self.option['flow'])

    def check_adj(self):
        H = (self.A - self.eigv[16]*self.B)
        H_adj = (self.C - np.conjugate(self.eigv[16])*self.E)  # (np.transpose(H))
        u = np.sin(np.arange(self.N))
        #pdb.set_trace()
        x = lin.solve(H, u)
        y = np.cos(np.arange(self.N))
        J1 = np.dot(y, x)
        v = lin.solve(H_adj, y)
        J2 = np.dot(v, u)

        print J1, J2
