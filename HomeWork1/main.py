import pylab as pl
import numpy as np
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftfreq

def rectangle_function(x, a):
    return 1.0*(abs(x)<a)

def recFourier(f, hw):
    return hw*np.sinc(2.0*hw*f)*2.0


if __name__  == "__main__":

    # Definition of general conditions of margins and time (may  be changed)
    left_limit = -10
    right_limit = 10
    dt = 0.001
    t = np.arange(left_limit, right_limit, dt)

    # Definition of rectangle
    a = 0.01 # This may be changed
    rec = rectangle_function(t, a)

    freq = np.fft.fftshift(np.fft.fftfreq(t.size, d = dt))

    recFFT = abs(np.fft.fftshift(np.fft.fft(rec)))
    FT = abs(recFourier(freq, a))

    # pl.figure(2)
    pl.plot(freq, recFFT / max(recFFT), linestyle = ':')
    pl.plot(freq, FT / max(FT))
    pl.xlim((-100,100))
    pl.show()
