import scipy as sp
import pylab as pl

if __name__ == '__main__':

    x = sp.linspace(0.0, 90, 1000)
    y_real = 20000 -(100 + x)**2
    y_imag = (100 + x)**2

    pl.figure()
    pl.xlabel('Distance to external border')
    pl.ylabel('Potential')
    pl.plot(x, y_real, label = 'real')
    pl.plot(x, y_imag, label = 'imaginary')
    pl.legend()
    pl.show()