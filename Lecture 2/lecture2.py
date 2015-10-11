__author__ = 'JoaoCosta'

import pylab as pl
from scipy.integrate import odeint

def fe1(x,t):
    gama = 0.1
    return pl.array([x[1], -(x[0] + gama * x[1])])

def e1(N, Tmax):
    time = pl.linspace(0.0, Tmax, N)
    xi = pl.array([0.0002, 0.2])
    x = odeint(fe1, xi, time)
    return x


def fe2(x,h):
    gama = 0.1
    return pl.array([x[0] + h * x[1], x[1] + h * (-x[0] - gama * x[1])])

def e2(N, Tmax):
    xi = xi = pl.array([0.0002, 0.2])
    h = Tmax / N
    x = [xi]
    for i in range(N-1):
        xi = fe2(xi, h)
        x.append(xi)
    return pl.array(x)

N_list = [2, 10, 100, 1000, 10000, 100000]
i_f = 1
last_ec = []
last_ode = []
for N in N_list:
    xode = e1(N, 10.0)
    x_ec = e2(N, 10.0)
    error = pl.absolute(xode[:,0]-x_ec[:,0])
    last_ode.append(xode[-1,0])
    last_ec.append(x_ec[-1,0])

    pl.figure(i_f)
    pl.plot(xode[:,0])
    pl.plot(x_ec[:,0])
    pl.plot(error)
    pl.title('N = ' + str(N))
    i_f +=1
print 'ODEINT'
print last_ode
print 'EULER'
print last_ec
pl.show()