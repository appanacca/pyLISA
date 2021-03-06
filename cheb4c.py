# from __future__ import division
import numpy as np
from scipy.linalg import toeplitz

def cheb4c(N):
	I = np.eye(N-2)  # Identity matrix
	n1 = N/2 -1
	n2 = int(round(N/2. -1))  # Indices used for flipping trick

	k = np.arange(N)  # Compute theta vector
	k = k[1:N-1]
	th = k*np.pi/(N-1)


	x = np.sin(np.pi*((N-3)-2*np.linspace(N-3,0,N))/(2*(N-1)))  # Compute Chebyshev points in the way W&R did it, with sin function.
	x = x[::-1]

	s = np.sin(th[0:n1+1]
			np.flipud(np.sin(th[0:n2+1]))

    T = np.tile(th/2,(N,1))
    DX = 2*np.sin(T.T+T)*np.sin(T.T-T)               # Trigonometric identity.
    DX[n1:,:] = -np.flipud(np.fliplr(DX[0:n2,:]))    # Flipping trick.!!!
    DX[range(N),range(N)]=1.                    # Put 1's on the main diagonal of DX.
