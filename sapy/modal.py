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
import scipy.sparse.linalg as lins
import scipy.interpolate as intp

import scipy.io

import blasius as bl
import numba as nb

import bokeh.plotting as bkpl
import bokeh.models as bkmd

import pdb as pdb


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

    """@classmethod
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
        return fluid(option) """

    def diff_matrix(self):
        """build the differenziation matrix with chebichev discretization
        [algoritmh from Reddy & W...]"""
        # in this line we re-instanciate the y in gauss lobatto points
        self.y, self.D = cb.chebdif(self.N, 4)
        self.D = self.D + 0j
        # summing 0j is needed in order to make the D matrices immaginary

    def read_velocity_profile(self):
        """ read from a file the velocity profile store in a
        .txt file and set the variable_data members"""
        in_txt = np.genfromtxt(self.option['flow'], delimiter=' ', skiprows=1)
        self.y_data = in_txt[:, 0]
        self.U_data = in_txt[:, 1]
        self.dU_data = in_txt[:, 2]
        self.ddU_data = in_txt[:, 3]
        self.aCD_data = in_txt[:, 4]
        self.daCD_data = in_txt[:, 5]
        self.lc = self.option['lc']  # 0.16739  #lc* = 0.22*(h-z1) / h

    def set_poiseuille(self):
        """set the members velocity and its derivatives as couette flow"""
        Upoiseuille = (lambda y: 1-y**2)
        dUpoiseuille = (lambda y: -y*2)
        ddUpoiseuille = -np.ones(len(self.y))*2
        self.U = Upoiseuille(self.y)
        self.dU = dUpoiseuille(self.y)
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

    def set_blasisus(self, y_gl):
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
                    (D4 - 2*self.alpha**2 * D2 + self.alpha**4 * I)\
                    - ddU + U*delta
            self.B = delta
        elif self.option['equation'] == 'LNS_CD':
            self.A = (i/(self.alpha*self.Re)) *\
                    (D4 - 2 * self.alpha**2 * D2 + self.alpha**4 * I)\
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
        # selector = np.isfinite(self.eigv)
        # self.eigv = self.eigv[selector]
        # self.eigf = self.eigf[:, selector]

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
        AA2 = i*self.alpha*I
        AA3 = self.D[0]

        AB1 = i*self.alpha*I

        if (self.option['equation'] == 'Euler_wave' or
                self.option['equation'] == 'Euler'):
                    AB2 = i*self.alpha*np.diag(self.U)
                    AC3 = i*self.alpha*np.diag(self.U)

        elif (self.option['equation'] == 'Euler_CD' or
              self.option['equation'] == 'Euler_CD_wave'):
                AB2 = i*self.alpha*np.diag(self.U) + np.diag(self.aCD*self.U)
                AC3 = + i*self.alpha*np.diag(self.U)

        elif self.option['equation'] == 'Euler_CD_turb':
            AB2 = (i*self.alpha*np.diag(self.U)-(2*self.lc**2) *
                   (np.dot(np.diag(self.dU),
                    self.D[1]) + np.dot(self.D[0], np.diag(self.ddU))))
            AC3 = + i*self.alpha*np.diag(self.U)

        elif self.option['equation'] == 'LNS':
            AB2 = i*self.alpha*np.diag(self.U) - delta/self.Re
            AC3 = i*self.alpha*np.diag(self.U) - delta/self.Re

        elif self.option['equation'] == 'LNS_CD':
            AB2 = (i*self.alpha*np.diag(self.U) - delta/self.Re +
                   np.diag(self.aCD*self.U))
            AC3 = + i*self.alpha*np.diag(self.U) - delta/self.Re

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

        AB3 = np.diag(self.dU)

        AC1 = self.D[0]
        AC2 = np.zeros((self.N, self.N))

        BA1 = BA2 = BA3 = BB1 = BB3 = BC1 = BC2 = np.zeros((self.N, self.N))
        BB2 = BC3 = i * I * self.alpha

        AA = np.concatenate((AA1, AA2, AA3), axis=1)
        AB = np.concatenate((AB1, AB2, AB3), axis=1)
        AC = np.concatenate((AC1, AC2, AC3), axis=1)

        self.A = np.concatenate((AA, AB, AC))

        BA = np.concatenate((BA1, BA2, BA3), axis=1)
        BB = np.concatenate((BB1, BB2, BB3), axis=1)
        BC = np.concatenate((BC1, BC2, BC3), axis=1)

        self.B = np.concatenate((BA, BB, BC))

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

    def adjoint_spectrum_v_eta(self):
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

        # self.C = 2 * dU * D1 + U*delta + (i/self.alpha)*(dCD*U*D1 + CD*dU*D1 + CD*U*D2)
        # self.E = delta
        self.C = np.conjugate(np.transpose(self.A))
        self.E = np.conjugate(np.transpose(self.B))
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

        elif self.option['variables'] == 'p_u_v':

            idx_bc = np.array([2*self.N, 3*self.N - 1])
            self.C[idx_bc, :] = np.zeros(3*self.N)
            self.E[idx_bc, :] = np.zeros(3*self.N)

            self.C[2*self.N, 2*self.N] = 1
            self.C[3*self.N - 1, 3*self.N - 1] = 1


    @nb.jit
    def solve_eig_adj(self):
        """ solve the eigenvalues problem with the LINPACK subrutines"""
        self.eigv_adj, self.eigf_adj = lin.eig(self.C, self.E)

        # self.eigv_adj, self.eigf_adj = lins.eigs(self.C, k=10, M=self.E)
        # doesent work ARPACK problems with Arnoldi factorization

        # remove the infinite and nan eigenvectors, and their eigenfunctions
        # selector = np.isfinite(self.eigv_adj)
        # self.eigv_adj = self.eigv_adj[selector]
        # self.eigf_adj = self.eigf_adj[:, selector]

    @nb.jit
    def omega_alpha_curves(self, alpha_start, alpha_end, n_step):
        self.vec_alpha = np.linspace(alpha_start, alpha_end, n_step)
        self.vec_eigv_im = np.zeros(n_step)
        for i in np.arange(n_step):
            self.set_perturbation(self.vec_alpha[i], self.Re)
            self.choose_variables()
            self.solve_eig()
            # self.vec_eigv_im[i] = np.max(self.eigv_im)

            self.vec_eigv_im[i] = (self.vec_alpha[i] *
                                   np.max(self.eigv_im[self.eigv_im < 1]))

            # print self.eigv_im
        np.savez('euler_cd_turb', self.vec_alpha, self.vec_eigv_im)

        fig, ay = plt.subplots(dpi=150)
        lines = ay.plot(self.vec_alpha, self.vec_eigv_im, 'b', lw=2)
        ay.set_ylabel(r'$\omega_i$', fontsize=32)
        ay.set_xlabel(r'$\alpha$', fontsize=32)
        # lgd = ay.legend((lines), (r'$U$', r'$\delta U$', r'$\delta^2 U$'),
        #                 loc=3, ncol=3, bbox_to_anchor=(0, 1), fontsize=32)
        # ay.set_ylim([-1, 0.1])
        # ay.set_xlim([0, 1.8])
        ay.grid()
        # plt.tight_layout()
        fig.savefig('euler_cd_turb.png', bbox_inches='tight', dpi=150)
        plt.show(lines)

    def set_perturbation(self, a, Re):
        self.alpha = a
        self.Re = Re

    def superpose_spectrum(self, alpha_start, alpha_end, n_step):
        self.vec_alpha = np.linspace(alpha_start, alpha_end, n_step)

        sp_re = np.array([])
        sp_im = np.array([])

        bkpl.output_file("spectrum.html")

        TOOLS = "resize, crosshair, pan, wheel_zoom, box_zoom, reset,\
                    box_select, lasso_select, hover"

        p = bkpl.figure(plot_width=1000, plot_height=600, tools=TOOLS,
                        title="Superimposed spectrum", x_axis_label='c_r',
                        y_axis_label='c_i', x_range=(0.8, 0.96),
                        y_range=(-0.05, 0.1))

        COLORS = ['aqua', 'blue', 'fuchsia', 'gold', 'green', 'orange', 'red',
                  'sienna', 'yellow', 'lime']   # from css list
        j = -1
        for i in np.arange(n_step):
            j = j+1
            self.set_perturbation(self.vec_alpha[i], self.Re)
            self.choose_variables()
            self.solve_eig()
            # sp_re = np.concatenate((sp_re, self.eigv_re))
            # sp_im = np.concatenate((sp_im, self.eigv_im))
            sp_re = self.eigv_re.tolist()
            sp_im = self.eigv_im.tolist()
            if j > 9:
                j = j-10    # these with the above inizialization is needed for
            # the iteration in the COLOURS list

            p.circle(sp_re, sp_im, size=10, fill_color=COLORS[j])
        bkpl.show(p)

    def save_sim(self, file_name):
        np.savez(file_name, sim_param_keys=np.array(self.option.keys()),
                 sim_param_values=np.array(self.option.values(), dtype=object),
                 U=self.U, dU=self.dU, y=self.y,
                 ddU=self.ddU, aCD=self.aCD, daCD=self.daCD,
                 eigv=self.eigv, eigf=self.eigf, D=self.D,
                 adj_eigv=self.eigv_adj, adj_eigf=self.eigf_adj)