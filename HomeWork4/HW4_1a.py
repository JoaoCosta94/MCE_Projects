__author__ = 'JoaoCosta'

import scipy as sp
from scipy.sparse import linalg
from scipy import sparse
import pylab as pl
from time import time
import platform

def potential_well(X, Y, x0, y0, a, b, v0):
    """
    This function generates the potential well
    """
    V = sp.zeros(X.shape)
    indexes = sp.where(((X-x0)/a)**2 + ((Y-y0)/b)**2 > 1.0)
    V[indexes] = v0
    return V

def absorving_borders_box(X, Y, xyT, xyMax, vM):
    """
    This function generates the absorbing potential on the borders
    """
    border = sp.zeros(X.shape, dtype = complex)
    idx = sp.where(abs(X) > (xyMax - xyT))
    idy = sp.where(abs(Y) > (xyMax - xyT))
    border[idx] += vM * 1j * (abs(X[idx]) - xyMax + xyT)**2
    border[idy] += vM * 1j * (abs(Y[idy]) - xyMax + xyT)**2
    return border

def lap(shape, spacing):
    """
    This function generates the laplacian operator
    :param shape:
    :param spacing:
    :return:
    """
    n = shape[0]*shape[1]
    return ( -4.*sparse.eye(n, n, 0) + sparse.eye(n, n, 1) + sparse.eye(n, n, -1) + sparse.eye(n, n, shape[1]) + sparse.eye(n, n, -shape[1]) )/spacing**2

if __name__ == '__main__':

    if (platform.system() == 'Windows'):
        folder = 'E1a_Results\\'
    else:
        folder = 'E1a_Results/'

    # Problem definition
    v0_array = sp.array([1e1, 1e2, 1e3, 1e3, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16])

    vM = 300.0
    a = 1.0
    b = 1.0
    x0 = 0.0
    y0 = 0.0

    # Box
    xyMin = -3.0
    xyMax = 3.0
    xyT = 2.0*xyMax/3.0
    dxy = 0.03
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    eList = []
    sList = []

    for v0 in v0_array:
        print v0
        V = potential_well(X, Y, x0, y0, a, b, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
        L = lap(X.shape, dxy)

        H = -L + sparse.diags(V.ravel(),0, format = 'dia')

        start = time()
        energies, states = linalg.eigsh(H, which = 'SM', k=1)

        eList.append(energies[0])

        sList.append(states[0])
        sp.save(folder + 'S_V0_'+str(v0)+'.npy', states[0])

    eList = sp.array(eList)

    pl.figure('Smallest Energy')
    pl.title('Smallest Energy')
    pl.scatter(v0_array, eList)
    pl.xlabel('V0')
    pl.ylabel('First energy')
    pl.show()