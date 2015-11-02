import numpy as np
from scipy import ndimage
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
    return np.exp(-1.0 * (x*x + y*y) + 0.5 * x * y)*((17.0*x*x - 16.0*x*y + 17.0*y*y - 20.0*np.pi*np.pi - 16.0)*np.sin(np.pi*(x + 2.0*y)) - 4.0*np.pi*(2.0*x + 7.0*y)*np.cos(np.pi*(x + 2.0*np.pi)))/4.0

def ravelIdx(idx, shape):
    idx = [shape[i] + idx[i] if idx[i] < 0 else idx[i] for i in range(len(idx))]
    I = 0
    dim = 1
    for i in range(-1, -len(idx) - 1, -1):
        I += idx[i]*dim
        dim *= shape[i]
    return I

def laplacianMatrix(shape):
    L = np.zeros((shape[0]*shape[1], shape[0]*shape[1]), dtype = np.int8)

    for i in range(1, shape[0] - 1):
        for j in range(1, shape[1] - 1):
            L[ravelIdx((i, j), shape), ravelIdx((i, j), shape)] -= 4
            L[ravelIdx((i, j), shape), ravelIdx((i, j - 1), shape)] += 1
            L[ravelIdx((i, j), shape), ravelIdx((i, j + 1), shape)] += 1
            L[ravelIdx((i, j), shape), ravelIdx((i - 1, j), shape)] += 1
            L[ravelIdx((i, j), shape), ravelIdx((i + 1, j), shape)] += 1

    for i in range(1, shape[0] - 1):
        L[ravelIdx((i, 0), shape), ravelIdx((i - 1, 0), shape)] += 1
        L[ravelIdx((i, 0), shape), ravelIdx((i + 1, 0), shape)] += 1

        L[ravelIdx((i, 0), shape), ravelIdx((i, 1), shape)] -= 5
        L[ravelIdx((i, 0), shape), ravelIdx((i, 2), shape)] += 4
        L[ravelIdx((i, 0), shape), ravelIdx((i, 3), shape)] -= 1


        L[ravelIdx((i, -1), shape), ravelIdx((i - 1, -1), shape)] += 1
        L[ravelIdx((i, -1), shape), ravelIdx((i + 1, -1), shape)] += 1

        L[ravelIdx((i, -1), shape), ravelIdx((i, -2), shape)] -= 5
        L[ravelIdx((i, -1), shape), ravelIdx((i, -3), shape)] += 4
        L[ravelIdx((i, -1), shape), ravelIdx((i, -4), shape)] -= 1


    for j in range(1, shape[1] - 1):
        L[ravelIdx((0, j), shape), ravelIdx((0, j - 1), shape)] += 1
        L[ravelIdx((0, j), shape), ravelIdx((0, j + 1), shape)] += 1

        L[ravelIdx((0, j), shape), ravelIdx((1, j), shape)] -= 5
        L[ravelIdx((0, j), shape), ravelIdx((2, j), shape)] += 4
        L[ravelIdx((0, j), shape), ravelIdx((3, j), shape)] -= 1


        L[ravelIdx((-1, j), shape), ravelIdx((-1, j - 1), shape)] += 1
        L[ravelIdx((-1, j), shape), ravelIdx((-1, j + 1), shape)] += 1

        L[ravelIdx((-1, j), shape), ravelIdx((-2, j), shape)] -= 5
        L[ravelIdx((-1, j), shape), ravelIdx((-3, j), shape)] += 4
        L[ravelIdx((-1, j), shape), ravelIdx((-4, j), shape)] -= 1


    L[ravelIdx(( 0,  0), shape), ravelIdx(( 0,  0), shape)] += 4
    L[ravelIdx(( 0,  0), shape), ravelIdx(( 0,  1), shape)] -= 5
    L[ravelIdx(( 0,  0), shape), ravelIdx(( 1,  0), shape)] -= 5
    L[ravelIdx(( 0,  0), shape), ravelIdx(( 0,  2), shape)] += 4
    L[ravelIdx(( 0,  0), shape), ravelIdx(( 2,  0), shape)] += 4
    L[ravelIdx(( 0,  0), shape), ravelIdx(( 0,  3), shape)] -= 1
    L[ravelIdx(( 0,  0), shape), ravelIdx(( 3,  0), shape)] -= 1

    L[ravelIdx(( 0, -1), shape), ravelIdx(( 0, -1), shape)] += 4
    L[ravelIdx(( 0, -1), shape), ravelIdx(( 0, -2), shape)] -= 5
    L[ravelIdx(( 0, -1), shape), ravelIdx(( 1, -1), shape)] -= 5
    L[ravelIdx(( 0, -1), shape), ravelIdx(( 0, -3), shape)] += 4
    L[ravelIdx(( 0, -1), shape), ravelIdx(( 2, -1), shape)] += 4
    L[ravelIdx(( 0, -1), shape), ravelIdx(( 0, -4), shape)] -= 1
    L[ravelIdx(( 0, -1), shape), ravelIdx(( 3, -1), shape)] -= 1

    L[ravelIdx((-1,  0), shape), ravelIdx((-1,  0), shape)] += 4
    L[ravelIdx((-1,  0), shape), ravelIdx((-1,  1), shape)] -= 5
    L[ravelIdx((-1,  0), shape), ravelIdx((-2,  0), shape)] -= 5
    L[ravelIdx((-1,  0), shape), ravelIdx((-1,  2), shape)] += 4
    L[ravelIdx((-1,  0), shape), ravelIdx((-3,  0), shape)] += 4
    L[ravelIdx((-1,  0), shape), ravelIdx((-1,  3), shape)] -= 1
    L[ravelIdx((-1,  0), shape), ravelIdx((-4,  0), shape)] -= 1

    L[ravelIdx((-1, -1), shape), ravelIdx((-1, -1), shape)] += 4
    L[ravelIdx((-1, -1), shape), ravelIdx((-1, -2), shape)] -= 5
    L[ravelIdx((-1, -1), shape), ravelIdx((-2, -1), shape)] -= 5
    L[ravelIdx((-1, -1), shape), ravelIdx((-1, -3), shape)] += 4
    L[ravelIdx((-1, -1), shape), ravelIdx((-3, -1), shape)] += 4
    L[ravelIdx((-1, -1), shape), ravelIdx((-1, -4), shape)] -= 1
    L[ravelIdx((-1, -1), shape), ravelIdx((-4, -1), shape)] -= 1

    return L

def matrixMethod(G, h):
    """
    This function calculates the laplacian of f(X,Y) with a matrix for the laplacian operator approach
    :param G:   f(X,Y)
    :param h:   Derivative step
    :return:    Laplacian of f(X,Y) using a matrix
    """
    oShape = G.shape
    lapM = laplacianMatrix(oShape)
    start = time.time()
    mul = np.dot(lapM, G.ravel())
    mTime = time.time() - start
    return mul.reshape(oShape)/(h*h), mTime

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

    # Matrix method result
    matrix_result, mTime = matrixMethod(fXY, h)
    matrix_error = matrix_result - analytical

    # Print of computation times
    print "Roll method took " + str(rTime) + " seconds"
    print "Convolution method took " + str(cTime) + " seconds"
    print "Matrix method took " + str(mTime) + " seconds"

    ##############################################

    # Plot of f(X,Y)
    pl.figure("Original function")
    pl.title("f(x,y)")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, fXY)
    pl.colorbar()

    # Plotting results
    min_level = min(analytical.ravel())
    max_level = max(analytical.ravel())
    n_levels = 1000
    levels = np.linspace(min_level, max_level, n_levels)

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

    pl.subplot(224)
    pl.title("Matrix")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, matrix_result, levels = levels)
    pl.colorbar()

    pl.subplots_adjust(hspace = 0.5, wspace = 0.5)

    # Error Plots
    # min_level = min([np.min(roll_error), np.min(conv_error), np.min(matrix_error)])
    # max_level = max([np.max(roll_error), np.max(conv_error), np.max(matrix_error)])
    min_level = -20.0
    max_level = 20.0
    n_levels = 100
    levels = np.linspace(min_level, max_level, n_levels)

    pl.figure("Results Error")

    pl.subplot(221)
    pl.title("Roll")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, roll_error, levels = levels)
    pl.colorbar()

    pl.subplot(222)
    pl.title("Convolution")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, conv_error, levels = levels)
    pl.colorbar()

    pl.subplot(223)
    pl.title("Matrix")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, matrix_error, levels = levels)
    pl.colorbar()

    pl.subplots_adjust(hspace = 0.5, wspace = 0.5)

    pl.show()