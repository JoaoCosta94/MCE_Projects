import numpy as np
import pylab as pl
import time

import HW1_2 as hw1

def f(x,y):
    """
    This function calculates the value of f on the points (x,y)
    :param x:   x coordinate (float or numpy array)
    :param y:   y coordinate (float or numpy array)
    :return:    f(x,y) (float or numpy array)
    """
    return np.sin(np.pi*(x + 2.0 * y)) * np.exp(-1.0 * (x*x + y*y) + 0.5 * x * y)

def analyticalLapf(x, y):
    """
    This function calculates the analytical laplacian of f on the points (x,y)
    :param x:   x coordinate (float or numpy array)
    :param y:   y coordinate (float or numpy array)
    :return:    lap(f) (float or numpy array)
    """
    return np.exp(-1.0 * (x*x + y*y) + 0.5 * x * y)*((17.0*x*x - 16.0*x*y + 17.0*y*y - 20.0*np.pi*np.pi - 16.0)*np.sin(np.pi*(x + 2.0*y)) - 4.0*np.pi*(2.0*x + 7.0*y)*np.cos(np.pi*(x + 2.0*np.pi)))

def matrixMethod(G):
    return "cenas"

if __name__ == '__main__':

    # Grid initialization
    xLim = 1.0
    yLim = 1.0
    nPoints = 128

    h = 2.0 * xLim / float(nPoints)
    X, Y  = np.mgrid[-xLim:xLim+h:h, -yLim:yLim+h:h]

    # Function f values on points (X,Y)
    fXY = f(X,Y)

    # Analytical result for laplacian of f(X,Y)
    start = time.time()
    analytical = analyticalLapf(X,Y)
    aTime = time.time() - start

    # Roll result (using implementation from 1st homework)
    roll_result, rTime = hw1.rollMethod(fXY, h)
    roll_error = roll_result - analytical

    # Convolution result (using implementation from 1st homework)
    conv_result, cTime = hw1.convMethod(fXY, h)
    conv_error = conv_result - analytical

    # # Matrix method result
    # start = time.time()
    # matrix_result = matrixMethod(fXY)
    # mTime = time.time() - start
    # matrix_error = matrix_result - analytical

    ##############################################

    min_level = min(analytical.ravel())
    max_level = max(analytical.ravel())
    n_levels = 1000
    levels = np.linspace(min_level, max_level, n_levels)

    # Plotting results
    pl.figure("Laplacian Results")
    pl.subplot(221)
    pl.title("Analytical")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, analytical, levels = levels)
    pl.colorbar()

    pl.subplot(222)
    pl.title("Roll")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, roll_result, levels = levels)
    pl.colorbar()

    pl.subplot(223)
    pl.title("Convolution")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, conv_result, levels = levels)
    pl.colorbar()

    # pl.subplot(224)
    # pl.title("Matrix")
    # pl.xlabel("x")
    # pl.ylabel("y")
    # pl.contourf(X, Y, matrix_result, levels = levels)
    # pl.colorbar()

    pl.subplots_adjust(hspace = 0.5, wspace = 0.5)

    pl.show()