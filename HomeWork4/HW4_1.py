__author__ = 'JoaoCosta'

import scipy as sp
from scipy.sparse import linalg
from scipy import sparse
import pylab as pl
from time import time

def potential_well(X, Y, x0, y0, a, b, v0):
    """
    This function generates the potential well
    :param X:
    :param Y:
    :param x0:
    :param y0:
    :param a:
    :param b:
    :param v0:
    :return:
    """
    V = sp.zeros(X.shape)
    indexes = sp.where(((X-x0)/a)**2 + ((Y-y0)/b)**2 > 1.0)
    V[indexes] = v0
    return V

def absorving_borders_box(X, Y, xyL, vM):
    """
    This function gerenrates the absorving potential on the borders
    :param X:
    :param Y:
    :param xyL:
    :param vM:
    :return:
    """
    x = abs(X)
    y = abs(Y)
    B = sp.zeros(X.shape, dtype = complex)
    id = sp.where(x > x - xyL)
    B[id] += (x[id] - (x[id]-xyL))**2
    id = sp.where(y > y - xyL)
    B[id] += (y[id] - (y[id]-xyL))**2
    return 1j*vM*B

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

    # Problem definition
    v0 = 100.0
    delta = sp.linspace(0.0, 1.0, 10)
    b = 1.0
    x0 = 0.0
    y0 = 0.0

    # Box
    xyMin = -2.5
    xyMax = 2.5
    xyL = 1.0
    dxy = 0.02
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    e1List = []
    e2List = []
    e3List = []

    s1List = []
    s2List = []
    s3List = []

    for d in delta:
        print d

        a = 1.0 + d
        V = potential_well(X, Y, x0, y0, a, b, v0) + absorving_borders_box(X, Y, xyL, 200)
        L = lap(X.shape, dxy)

        H = -L + sparse.diags(V.ravel(),0, format = 'dia')

        energies, states = linalg.eigsh(H, which = 'SM', k=3)

        e1List.append(energies[0])
        e2List.append(energies[1])
        e3List.append(energies[2])
        sp.save('E1_e_d_'+str(d)+'.npy', energies)

        s1List.append(states[0])
        s2List.append(states[1])
        s3List.append(states[2])
        sp.save('E1_s1_d_'+str(d)+'.npy', states[0])
        sp.save('E1_s2_d_'+str(d)+'.npy', states[1])
        sp.save('E1_s3_d_'+str(d)+'.npy', states[2])

    pl.figure('energies')
    pl.scatter(delta, e1List, label = 'first state energy')
    pl.scatter(delta, e2List, label = 'second state energy')
    pl.scatter(delta, e3List, label = 'third state energy')
    pl.legend()
    pl.show()