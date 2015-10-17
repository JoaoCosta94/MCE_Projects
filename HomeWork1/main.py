import pylab as pl
import numpy as np
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftfreq

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
    # rec = np.zeros(x.shape)
    #
    # # generating the actual rectangle function
    # # finding the start index of the rectangle
    # try:
    #     index_left = x.tolist().index(left_limit)
    # except:
    #     for i in range(len(x)):
    #         if (x[i] >= left_limit - spacing) and (x[i] <= left_limit + spacing):
    #             index_left = i
    # # finding the right limit index
    # index_right = index_left + w / spacing
    # heights = h*np.ones((w / spacing + 1,))
    #
    # # rearranging the rectangle
    # rec[index_left:index_right+1] = heights
    # return rec
    return 1.0*(abs(x)<-left_limit)

def fourier_numerical(signalsList, spacing):
    """
    This method calculates given signals FFT using numpy.fft
    :param signalsList: List with signals array to transform
    :return: Tuple with list of FFT of original signals and list of FFT frequencies
    """
    signalsFFT = []
    fftFreqs = []
    for signal in signalsList:
        signalsFFT.append(fft(signal))
        fftFreqs.append(fftfreq(signal.size, d=spacing))

    return (signalsFFT, fftFreqs)

def recFourier(f, hw):
    return hw*np.sinc(hw*f)*2.0

def cosFourier(f, w):
    cF = np.zeros((len(f),))
    for i in range(len(f)):
        if (abs(f[i]-dt) <= w) and (abs(f[i]+dt) >= w):
            print "ta bom caralho"
            cF[i] = 1.0
    return cF

if __name__  == "__main__":

    # Definition of general conditions of margins and time (may be changed)
    left_limit = -10
    right_limit = 10
    dt = 0.001
    t = np.arange(left_limit, right_limit, dt)
    f = np.linspace(-0.5/dt, 0.5/dt, len(t))

    # Definition of rectangle
    a = 0.5 # This may be changed
    rec = rectangle_function(t, dt, -a, 2*a, 1)
    Prec = sum(rec * rec)

    # Definition of original cosine function
    w = 1 # This may be changed
    c = np.cos(w * t)
    Pc = sum(c * c)

    # FFT of original signals
    # Analytical solution
    recFT = recFourier(f * 2.0*np.pi, a)
    PrecFT = sum(recFT * recFT)
    recFT = abs(recFT) * Prec / PrecFT

    # cFT = [-(w+dt) + i*dt for i in range(3)] + [(w-dt) + i*dt for i in range(3)]
    # cFT = cosFourier(cFT, w)
    cFT = cosFourier(f * 2.0*np.pi, w)
    print cFT
    PcFT = sum(cFT * cFT)
    cFT = abs(cFT) * Pc / PcFT
    # Numerical solution
    fftRes, fftFreq = fourier_numerical((rec, c), dt)

    recFFT = fftRes[0]
    PrecFFT = sum(recFFT * recFFT)
    recFFT = abs(recFFT) * Prec / PrecFFT

    cFFT = fftRes[1]
    PcFFT = sum(cFFT * cFFT)
    cFFT = abs(cFFT) * Pc / PcFFT
    # Convolutions
    # By numpy.convolve
    convNP = np.convolve(rec, c, "same")
    # By convolution theorem
    conv = fftconvolve(rec, c, "same")


    # # Plotting of original signals
    # pl.figure(1)
    # pl.title("Original Signals")
    # pl.xlabel("time")
    # pl.ylabel("f(t)")
    # pl.plot(t,rec, label = "rec(t)")
    # pl.plot(t, c, label = "cos(wt)")
    # pl.legend()

    # Plotting FFTs
    pl.figure("Fourier Transforms")
    pl.subplot(221)
    pl.title("REC FFT")
    pl.plot(fftFreq[0], recFFT)
    pl.xlim(left_limit, right_limit)

    pl.subplot(223)
    pl.title("REC FT")
    pl.plot(f, recFT)
    pl.xlim(left_limit, right_limit)

    pl.subplot(222)
    pl.title("COS FFT")
    pl.plot(fftFreq[1], cFFT)
    pl.xlim((-5,5))

    pl.subplot(224)
    pl.title("COS FT")
    pl.plot(f, cFT)
    pl.xlim((-5,5))

    # # Plotting convolutions
    # pl.figure("Convolutions")
    # pl.subplot(211)
    # pl.title("Convolution by Numpy Convolve")
    # pl.plot(t, convNP)
    # pl.subplot(212)
    # pl.title("Convolution by FFTConvolve")
    # pl.plot(t, conv)

    pl.show()

