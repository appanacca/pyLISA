from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as intp
import scipy.integrate as integ
import scipy.optimize as opt


def blasius(y):
    nb_pts = 1e5
    eta_max = 10
    d_eta = eta_max / nb_pts
    eta = np.linspace(0, eta_max+d_eta, nb_pts)

    s = opt.brentq()


def bl_integ(s, eta):
    X0 = [0, 0, s]
    eta, X = integ.ode(bl_sys(eta, X), jac=bl_jac(eta,X)).set_integrator('dopri5', atol=1e-10,
            rtol=1e-10)


def bl_sys(eta, X):
    f = X[0]
    u = X[1]
    g = X[3]

    dX = np.array([np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))])
    dX[0] = u
    dX[1] = g
    dX[2] = -f*g

def bl_jac(eta, X):
    f = X[0]
    u = X[1]
    g = X[3]
    
    jac = np.array([[0, 1, 0], [0, 0, 1], [-g, 0, -f]])


