import numpy as np
from numpy import sin, cos, exp
from scipy.signal import convolve2d
import pylab as pl
import time

def f(x,y):
    """
    This function calculates the value of f on the point 8x,y)
    :param x: Point x coordinate
    :param y: Point y coordinate
    :return: Value of f on point (x,y)
    """
    return (np.sin(x) + np.sin(x+y) + np.sin(x*y))*np.exp(-(x*x + y*y)/sigma)

def analyticalLapf(x, y, o):
    """
    This function returns de analytical laplacian of function f
    :param X:
    :param Y:
    :return:
    """
    return (-x**2*sin(x*y) - sin(x + y))*exp((-x**2 - y**2)/o) + (-y**2*sin(x*y) - sin(x) - sin(x + y))*exp((-x**2 - y**2)/o) - 4*x*(y*cos(x*y) + cos(x) + cos(x + y))*exp((-x**2 - y**2)/o)/o - 4*y*(x*cos(x*y) + cos(x + y))*exp((-x**2 - y**2)/o)/o - 4*(sin(x) + sin(x*y) + sin(x + y))*exp((-x**2 - y**2)/o)/o + 4*x**2*(sin(x) + sin(x*y) + sin(x + y))*exp((-x**2 - y**2)/o)/o**2 + 4*y**2*(sin(x) + sin(x*y) + sin(x + y))*exp((-x**2 - y**2)/o)/o**2

def dx2For(G, h):
    """
    This function calculates dx2/(dx)^2 f(X,Y)
    :param G: Values of f(X,Y)
    :param h: Derivation step
    :return: dx2/(dx)^2 f(X,Y)
    """
    R = np.empty_like(G)
    h2 = h*h
    # The derivative on the X borders
    for k in range(len(G[0])):
        # Value of the derivative on the left border
        R[-1, k] = (-2.0*G[-1, k] + G[-2, k])/h2
        # Value of the derivative on the right border
        R[0, k] = (-2.0*G[0, k] + G[1, k])/h2
    # The "inside" points
    for i in range(1, len(G)-1):
        # All points that are not in the borders
        for j in range(len(G[0])):
            R[i, j] = (G[i+1, j] - 2.0*G[i,j] + G[i-1, j])/h2
    return R

def dy2For(G,h):
    """
    This function calculates dy2/(dy)^2 f(X,Y)
    :param G: Values of f(X,Y)
    :param h: Derivation step
    :return: dy2/(dy)^2 f(X,Y)
    """
    R = np.empty_like(G)
    h2 = h*h
    # The derivative on the borders
    for k in range(len(G)):
        # Value of the derivative on the top border
        R[k, -1] = (-2.0*G[k, -1] + G[k, -2])/h2
        # Value of the derivative on the bottom border
        R[k, 0] = (-2.0*G[k, 0] + G[k, 1])/h2
    # The "inside" points
    for i in range(len(G)):
        # All points that are not in the borders
        for j in range(1, len(G[0])-1):
            R[i, j] = (G[i, j+1] - 2.0*G[i,j] + G[i, j-1])/h2
    return R

def forMethod(G, h):
    """
    This function calculates the laplacian with a python for approach
    :param G:   f(X,Y)
    :param h:   Derivative step
    :return:    laplacian of f(X,Y) using for method
    """
    start = time.time()
    lap = dx2For(G,h) + dy2For(G,h)
    return (lap, time.time()-start)

def rollMethod(G,h):
    """
    This function calculates the laplacian of f(X,Y) with a numpy roll for approach
    :param G:   f(X,Y)
    :param h:   Derivative step
    :return:    Laplacian of f(X,Y) using numpy roll
    """
    start = time.time()
    dx2 = np.roll(G, 1, 0) -2.0*G + np.roll(G, -1, 0)
    dy2 = np.roll(G, 1, 1) -2.0*G + np.roll(G, -1, 1)
    return ((dx2 + dy2)/(h*h), time.time()-start)

def convMethod(G, h):
    """
    This function calculates the laplacian of f(X,Y) with a computational molecule and convolution approach
    :param G:   f(X,Y)
    :param h:   Derivative step
    :return:    Laplacian of f(X,Y) using a computational molecule
    """
    start = time.time()
    computationalMolecule = np.array([[0.0, 1.0, 0.0],
                                      [1.0, -4.0, 1.0],
                                      [0.0, 1.0, 0.0]])
    # Since on the other mehtods, on the boundaries, the neighbours were considered to be 0,
    # here the boundary was used the default mode which is fill, and the fill value being 0 (also default)
    return (convolve2d(G, computationalMolecule, mode = 'same')/(h*h), time.time() - start)

def fftMethod(G, h):
    """
    This function calculates the laplacian of f(X,Y) with a fourier space laplacian approach
    :param G:   f(X,Y)
    :param h:   Derivative step
    :return:    Laplacian of f(X,Y) using a FT
    """
    start = time.time()
    sizeX = len(G)
    sizeY = len(G[0])
    # Sampling frequencies
    fX, fY = np.meshgrid(np.fft.fftfreq(sizeX, d = h), np.fft.fftfreq(sizeY, d = h))
    # FFT2D of f(X,Y)
    transf2D = np.fft.fft2(G)
    # FT Laplacian = -(Wx^2 + Wy^2) * F(Wx, Wy)
    lapFourier = transf2D * (2.0*np.pi)**2 * (-(fX**2 + fY**2))
    # Laplacian = FT^-1(FT Laplacian)
    return (np.fft.ifft2(lapFourier).real, time.time() - start)

if __name__  == "__main__":
    # Initial conditions (Maybe changed)
    global sigma
    sigma = 3.0
    xLim = 5.0
    yLim = 5.0

    # Creation of list of derivative steps to evaluate (May be changed)
    h_list = np.linspace(0.001, 1.0, 100)
    # h_list = [0.01]
    result_for_list = []
    time_for_list = []
    error_for_list = []

    result_roll_list = []
    time_roll_list = []
    error_roll_list = []

    result_conv_list = []
    time_conv_list = []
    error_conv_list = []

    result_fft_list = []
    time_fft_list = []
    error_fft_list = []

    analyticalSol_list = []

    # Beginning of calculations
    start = time.time()
    for h in h_list:
        # Creation of the points grid to evaluate the second derivative
        X, Y = np.mgrid[-xLim:xLim+h:h, -yLim:yLim+h:h]
        # Calculation of f(X,Y)
        fXY = f(X, Y)
        # Calculation of the Laplacian through the analytical expression
        lapAnalytical = analyticalLapf(X,Y, sigma)
        analyticalSol_list.append(lapAnalytical)

        # Calculation through for method
        result, t = forMethod(fXY, h)
        result_for_list.append(result)
        time_for_list.append(t)
        error_for_list.append(np.average(abs(result - lapAnalytical)).ravel())

        # By numpy roll
        result, t = rollMethod(fXY, h)
        result_roll_list.append(result)
        time_roll_list.append(t)
        error_roll_list.append(np.average(abs(result - lapAnalytical)).ravel())

        # By convolution with computational molecule
        result, t = convMethod(fXY, h)
        result_conv_list.append(result)
        time_conv_list.append(t)
        error_conv_list.append(np.average(abs(result - lapAnalytical)).ravel())

        # By FT laplacian
        result, t = fftMethod(fXY, h)
        result_fft_list.append(result)
        time_fft_list.append(t)
        error_fft_list.append(np.average(abs(result - lapAnalytical)).ravel())

    print "Calculations took " + str(time.time() - start) + " seconds"

    start = time.time()
    # min_h = min(h_list)
    # j = h_list.index(min_h)
    # # Determination of color scale. Using the analytical solution for reference
    # min_level = min(analyticalSol_list[j].ravel())
    # max_level = max(analyticalSol_list[j].ravel())
    # n_levels = 1000
    # levels = np.linspace(min_level, max_level, n_levels)
    #
    # # Plot of Analytical solution
    # X, Y = np.mgrid[-xLim:xLim+min_h:min_h, -yLim:yLim+min_h:min_h]
    # pl.figure("Analytical Solution")
    # pl.title("Analytical Solution")
    # pl.xlabel("x")
    # pl.ylabel("y")
    # pl.contourf(X, Y, analyticalSol_list[j], levels = levels)
    # pl.colorbar()
    # # Plotting of results
    # for i in range(len(h_list)):
    #     X, Y = np.mgrid[-xLim:xLim+h_list[i]:h_list[i], -yLim:yLim+h_list[i]:h_list[i]]
    #     pl.figure("h = " + str(h_list[i]))
    #     # Plot of for method results
    #     pl.subplot(221)
    #     pl.title("For")
    #     pl.xlabel("x")
    #     pl.ylabel("y")
    #     pl.contourf(X, Y, result_for_list[i], levels = levels)
    #     pl.colorbar()
    #
    #     # Plot of roll method results
    #     pl.subplot(222)
    #     pl.title("Numpy Roll")
    #     pl.xlabel("x")
    #     pl.ylabel("y")
    #     pl.contourf(X, Y, result_roll_list[i], levels = levels)
    #     pl.colorbar()
    #
    #     # Plot of computational molecule method results
    #     pl.subplot(223)
    #     pl.title("Computational Molecule")
    #     pl.xlabel("x")
    #     pl.ylabel("y")
    #     pl.contourf(X, Y, result_conv_list[i], levels = levels)
    #     pl.colorbar()
    #
    #     # Plot of FT laplacian method result
    #     pl.subplot(224)
    #     pl.title("FT")
    #     pl.xlabel("x")
    #     pl.ylabel("y")
    #     pl.contourf(X, Y, result_fft_list[i], levels = levels)
    #     pl.colorbar()
    #
    #     pl.subplots_adjust(hspace = 0.5, wspace = 0.5)

    # Plot of smaller h 2D error
    # Preparation of variables
    i = np.argmin(h_list)
    minh_for = result_for_list[i] - analyticalSol_list[i]
    minh_roll = result_roll_list[i] - analyticalSol_list[i]
    minh_conv = result_conv_list[i] - analyticalSol_list[i]
    minh_fft = result_fft_list[i] - analyticalSol_list[i]
    levels = np.linspace(-8, 8, 5)
    # Actual plots
    pl.figure("Lalplacian Error")
    pl.subplot(221)
    pl.title("For")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X,Y, minh_for, levels = levels)
    pl.colorbar()
    pl.subplot(222)
    pl.title("Roll")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X,Y, minh_roll, levels = levels)
    pl.colorbar()
    pl.subplot(223)
    pl.title("Conv")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X,Y, minh_conv, levels = levels)
    pl.colorbar()
    pl.subplot(224)
    pl.title("FFT")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X,Y, minh_fft, levels = levels)
    pl.colorbar()
    pl.subplots_adjust(hspace = 0.5, wspace = 0.5)

    # Plot of statical variables
    pl.figure("Statistical analyses")
    pl.subplot(211)
    pl.plot(h_list, time_for_list, label = "For Method", ls = "", marker = '+')
    pl.plot(h_list, time_roll_list, label = "Roll Method", ls = "", marker = 'o')
    pl.plot(h_list, time_conv_list, label = "Conv Method", ls  = "", marker = 'D')
    pl.plot(h_list, time_fft_list, label = "FFT Method", ls = "", marker = 'p')
    pl.xlabel("h")
    pl.ylabel("Calculation Time (s)")
    pl.legend()
    pl.subplot(212)
    pl.plot(h_list, error_for_list, label = "For Method", ls = "", marker = '+')
    pl.plot(h_list, error_roll_list, label = "Roll Method", ls = "", marker = 'o')
    pl.plot(h_list, error_conv_list, label = "Conv Method", ls  = "", marker = 'D')
    pl.plot(h_list, error_fft_list, label = "FFT Method", ls = "", marker = 'p')
    pl.xlabel("h")
    pl.ylabel("Average Error")
    pl.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)

    print "Plotting graphs took " + str(time.time() - start) + " seconds"
    pl.subplots_adjust(hspace = 0.8)
    pl.show()