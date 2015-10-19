import numpy as np
from __future__ import division


def clencurt(N):
    """ CLENCURT  nodes x (Chebyshev points) and weights 
    for Clenshaw-Curtis quadrature"""
    
    theta = np.pi*np.arange(0,N+1) / N
    x = np.cos(theta)
    w = np.zeros(N+1) 
    ii = np.arange(1,N) 
    v = np.ones(N-1)
    if np.mod(N, 2) == 0:
        w[0] = 1/ (N**2 -1)
        w[-1] = w[0]
        for k in np.arange(1, N/2):
                v = v - 2*np.cos(2*k*theta[ii])/(4* k**2 -1)
        v = v - np.cos(N*theta[ii])/(N**2-1)
    else:
        w[0] = 1/ N**2
        w[-1] = w[0]
        for k in np.arange(1, N/2):
            v = v - 2*np.cos(2*k*theta[ii])/(4* k**2 -1) 
    w[ii] = 2*v/N
    return x,w
