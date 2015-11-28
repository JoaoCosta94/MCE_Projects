__author__ = 'JoaoCosta'

import scipy as sp
from scipy import special
from numpy import sum
import pylab as pl

def bess(n, R):
    """
    This function calculates the n-th order Bessel function
    :param n:   Bessel function order
    :param R:   Points to evaluate the function
    :return:    N-th order Bessel function on points R
    """
    return special.jv(n, R)

axis_points = sp.linspace(-2.0, 2.0, 1000)
spacing = abs(axis_points[1]-axis_points[0])
X ,Y = sp.meshgrid(axis_points, axis_points)

R = sp.sqrt(X**2 + Y**2)
theta = sp.arctan2(Y, X)
n_list = sp.array([0, 1])

# solving for first energy
fr = bess(0, R * special.jn_zeros(0, 1)[0])
gt = 1.0
phi = (fr * gt)**2
phi = phi*(R <= 1)
phi /= sum(phi) * spacing ** 2

pl.figure("First State")
pl.contourf(X,Y, phi, levels = sp.linspace(phi.min(), phi.max(), 100))
pl.colorbar()
pl.contour(X, Y, (R <= 1) * 1)

# solving for second energy
fr = bess(1, R * special.jn_zeros(1, 1)[0])
gt = sp.sin(theta)
phi = (fr * gt) ** 2
phi = phi*(R <= 1)
A = sum(phi) * spacing ** 2
phi /= sum(phi) * spacing ** 2

pl.figure("Second State")
pl.contourf(X,Y, phi, levels = sp.linspace(phi.min(), phi.max(), 100))
pl.colorbar()
pl.contour(X, Y, (R <= 1) * 1)

pl.show()

