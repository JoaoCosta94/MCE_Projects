import pylab as pl
import numpy as np
from scipy.signal import fftconvolve

def rectangle_function(x, spacing, left_limit, w, h):
    """
    This function generates a rectangle with user's desired properties
    :param x:           Points where rectangle will be calculated
    :param spacing:     Spacing between consecutive points
    :param left_limit:  Point where rectangle begins (viewing from -inf)
    :param w:           Rectangle width
    :param h:           Height of the rectangle
    :return:            Array with desired rectangle
    """
    # generates a array with nPoints zeros
    rec = np.zeros(x.shape)

    # generating the actual rectangle function
    # finding the start index of the rectangle
    try:
        index_left = x.tolist().index(left_limit)
    except:
        for i in range(len(x)):
            if (x[i] >= left_limit - spacing) and (x[i] <= left_limit + spacing):
                index_left = i
    # finding the right limit index
    index_right = index_left + w / spacing
    heights = h*np.ones((w / spacing + 1,))

    # rearranging the rectangle
    rec[index_left:index_right+1] = heights
    return rec

def fourier_numerical(signalsList, spacing):
    """
    This method calculates given signals FFT using numpy.fft
    :param signalsList: List with signals array to transform
    :return: Tuple with list of FFT of original signals and list of FFT frequencies
    """
    signalsFFT = []
    fftFreqs = []
    for signal in signalsList:
        signalsFFT.append(np.fft.fftshift(np.fft.fft(signal)))
        fftFreqs.append(np.fft.fftshift(np.fft.fftfreq(signal.size, d=spacing)))

    return (signalsFFT, fftFreqs)

def recFourier(f, hw):
    return hw*np.sinc(2.0*hw*f)*2.0

if __name__  == "__main__":

    # Definition of general conditions of margins and time (may be changed)
    left_limit = -10
    right_limit = 10
    dt = 0.01
    t = np.arange(left_limit, right_limit, dt)

    # Definition of rectangle
    a = 0.5 # This may be changed
    rec = rectangle_function(t, dt, -a, 2*a, 1)

    # Definition of original cosine function
    w = 1 # This may be changed
    c = np.cos(w * t)

    # FFT of original signals
    # Numerical solution
    fftRes, fftFreq = fourier_numerical((rec, c), dt)

    recFFT = abs(fftRes[0])
    cosFFT = abs(fftRes[1])
    freq = fftFreq[0]

    # Analytical solution
    recFourierTransform = abs(recFourier(freq, a))
    # missing cosine

    # Convolutions
    # By numpy.convolve
    convNP = np.convolve(rec, c, "same")
    # By fourier transform convolution
    conv = fftconvolve(rec, c, "same")


    # Plotting of original signals
    pl.figure("Original Signals")
    pl.subplot(211)
    pl.title("Rectangle")
    pl.xlabel("time")
    pl.ylabel("f(t)")
    pl.plot(t,rec, label = "rec(t)")
    pl.subplot(212)
    pl.title("Cosine")
    pl.xlabel("time")
    pl.ylabel("f(t)")
    pl.plot(t, c, label = "cos(wt)")

    # Plotting FFTs
    pl.figure("Fourier Transforms")
    # Rectangular Pulse Transform
    pl.subplot(211)
    pl.title("Rectangular Pulse Transform")
    pl.plot(freq, recFFT / max(recFFT), label = "FFT")
    pl.plot(freq, recFourierTransform / max(recFourierTransform), label = "Analytical Solution")
    pl.xlim((-50, 50))
    pl.legend()
    # Cosine Transform
    pl.subplot(212)
    pl.title("Cosine Transform")
    pl.plot(freq, cosFFT / max(cosFFT), label = "FFT")
    pl.xlim((-w/np.pi, w/np.pi))
    pl.legend()

    # Plotting convolutions
    pl.figure("Convolutions")
    pl.subplot(211)
    pl.title("Convolution Result")
    pl.plot(t, convNP, label = "Numpy Convolve")
    pl.plot(t, conv, label = " FFTConvolve")
    pl.xlabel("time")
    pl.ylabel("f(t)")
    pl.legend()
    pl.subplot(212)
    pl.title("Error")
    pl.plot(t, abs(conv-convNP))
    pl.xlabel("time")
    pl.ylabel("f(t)")
    pl.ylim((-10**(-12)), 10**(-12))

    pl.show()

