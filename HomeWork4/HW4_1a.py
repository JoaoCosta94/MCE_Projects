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

    # Creating or loading operator matrix
    if (platform.system() == 'Windows'):
        folder = 'E1a_Results\\'
    else:
        folder = 'E1a_Results/'

    # Problem definition
    v0 = 1e14
    vM = 300.0
    delta = sp.array([0.0])
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

    t = []
    for d in delta:

        a = 1.0 + d
        V = potential_well(X, Y, x0, y0, a, b, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
        L = lap(X.shape, dxy)

        H = -L + sparse.diags(V.ravel(),0, format = 'dia')

        print 'calculating'
        start = time()
        energies, states = linalg.eigsh(H, which = 'SM', k=1)
        t.append(time() - start)
        print 'done'

        eList.append(energies[0])

        sList.append(states[0])
        sp.save(folder + 'S_d_'+str(d)+'.npy', states[0])

    eList = sp.array(eList)
    t = sp.array(t)

    print eList

    # Saving delta, energies and times to file
    sp.save(folder+'Delta'+'.npy', delta)
    sp.save(folder+'E'+'.npy', eList)
    sp.save(folder+'T'+'.npy', t)

    # Plotting energy and time
    pl.figure('Energies')
    pl.title('Energy study with V0 = '+str(v0))
    pl.xlabel(r'$\delta$')
    pl.ylabel(r'E($\delta$)')
    pl.ylim()
    pl.scatter(delta, eList, marker = 'o', label = 'first state energy')
    pl.legend()

    pl.show()