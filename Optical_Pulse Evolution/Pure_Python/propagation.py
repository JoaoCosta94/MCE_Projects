from numpy import *
from pylab import *

dt = .001
dx = .01
eps = .1#dt/dx**2

x = arange(0., 100., dx)
state = exp(-(x - 50.)**2/10.) * exp(-1j*1000.*x)

plot(x, abs(state)**2 / (abs(state)**2).max())

for i in range(1000):
    state[1:-1] += 1j*eps*(state[2:] + state[:-2])
    state[0] = 0.
    state[-1] = 0.

plot(x, abs(state)**2 / (abs(state)**2).max())
show()