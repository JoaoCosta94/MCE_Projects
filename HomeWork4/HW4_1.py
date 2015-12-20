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
        folder = 'E1_Results\\'
    else:
        folder = 'E1_Results/'

    # Problem definition
    v0 = 100.0
    vM = 300.0
    delta = sp.linspace(0.0, 1.0, 20)
    b = 1.0
    x0 = 0.0
    y0 = 0.0

    # Box
    xyMin = -3.0
    xyMax = 3.0
    xyT = 2.0*xyMax/3.0
    dxy = 0.03
    X, Y = sp.mgrid[xyMin:xyMax:dxy, xyMin:xyMax:dxy]

    e1List = []
    e2List = []
    e3List = []

    s1List = []
    s2List = []
    s3List = []

    t = []
    for d in delta:
        print d

        a = 1.0 + d
        V = potential_well(X, Y, x0, y0, a, b, v0) + absorving_borders_box(X, Y, xyT, xyMax, vM)
        L = lap(X.shape, dxy)

        H = -L + sparse.diags(V.ravel(),0, format = 'dia')

        start = time()
        energies, states = linalg.eigsh(H, which = 'SM', k=3)
        t.append(time() - start)

        e1List.append(energies[0])
        e2List.append(energies[1])
        e3List.append(energies[2])
        # sp.save(folder + 'E_d_'+str(d)+'.npy', energies)

        s1List.append(states[0])
        s2List.append(states[1])
        s3List.append(states[2])
        sp.save(folder + 'S1_d_'+str(d)+'.npy', states[0])
        sp.save(folder + 'S2_d_'+str(d)+'.npy', states[1])
        sp.save(folder + 'S3_d_'+str(d)+'.npy', states[2])

    e1List = sp.array(e1List)
    e2List = sp.array(e2List)
    e3List = sp.array(e3List)
    t = sp.array(t)

    # Saving delta, energies and times to file
    sp.save(folder+'Delta'+'.npy', delta)
    sp.save(folder+'E1'+'.npy', e1List)
    sp.save(folder+'E2'+'.npy', e2List)
    sp.save(folder+'E3'+'.npy', e3List)
    sp.save(folder+'T'+'.npy', t)

    # Plotting energy and time
    pl.figure('Energies')
    pl.title('Energy study with V0 = '+str(v0))
    pl.xlabel(r'$\delta$')
    pl.ylabel(r'E($\delta$)')
    pl.ylim()
    pl.scatter(delta, e1List, marker = 'o', label = 'first state energy')
    pl.scatter(delta, e2List, marker = 'v', label = 'second state energy')
    pl.scatter(delta, e3List, marker = '*', label = 'third state energy')
    pl.legend()

    pl.figure('Time')
    pl.title('Time study with V0 = '+str(v0))
    pl.xlabel(r'$\delta$')
    pl.ylabel(r'T($\delta$)')
    pl.scatter(delta, t)
    pl.show()