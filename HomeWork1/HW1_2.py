import numpy as np
import pylab as pl

def f(x,y):
    """
    This function calculates the value of f on the point 8x,y)
    :param x: Point x coordinate
    :param y: Point y coordinate
    :return: Value of f on point (x,y)
    """
    return (np.sin(x) + np.sin(x+y) + np.sin(x*y))*np.exp(-(x*x + y*y)/sigma)

def analyticalLapf(X,Y):
    """
    This function returns de analytical laplacian of function f
    :param X:
    :param Y:
    :return:
    """
    return "cenas"

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
    :param X:
    :param Y:
    :param h:
    :return:
    """
    return "cenas"

def rollMethod(X,Y,h):
    return "cenas"

def convMethod(X,Y,h):
    return "cenas"

if __name__  == "__main__":
    # Initial conditions (Maybe changed)
    global sigma
    sigma = 3.0
    xLim = 5.0
    yLim = 5.0
    spacing = 0.1

    # Creation of the points grid to evaluate the second derivative
    X, Y = np.mgrid[-xLim:xLim+spacing:spacing, -yLim:yLim+spacing:spacing]
    fV = f(X, Y)
    print fV
    # Calculation of the Laplacian through the analytical expression
    lapAnalytical = analyticalLapf(X,Y)

    # Creation of list of derivative steps to evaluate
    h_list = [1, 0.1, 0.01, 0.001, 0.0001]
    result_for_list = []
    time_for_list = []
    error_for_list = []

    result_roll_list = []
    time_roll_list = []
    error_roll_list = []

    result_conv_list = []
    time_conv_list = []
    error_conv_list = []
    for h in h_list:
        # Calculation through for method
        result, time = forMethod(fV, h)
        result_for_list.append(result)
        time_for_list.append(result)
        # error_for_list.append(abs(result - lapAnalytical))
    #
    #     # By numpy roll
    #     result, time = rollMethod(fV, h)
    #     result_roll_list.append(result)
    #     time_roll_list.append(result)
    #     # error_roll_list.append(abs(result - lapAnalytical))
    #
    #     # By convolution with computational molecule
    #     result, time = convMethod(fV, h)
    #     result_conv_list.append(result)
    #     time_conv_list.append(result)
    #     # error_conv_list.append(abs(result - lapAnalytical))