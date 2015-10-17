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
    """
    This function calculates the analytical Fourier Transform of a rectangular pulse
    :param f:   frequencies where to calculate de fourier transform
    :param hw:  Half width of the rectangular pulse
    :return:    Array with fourier transform of given rectangular pulse
    """
    return hw*np.sinc(2.0*hw*f)*2.0

if __name__  == "__main__":

    # Definition of general conditions of margins and time (may be changed)
    left_limit = -10
    right_limit = 10
    dt = 0.01
    t = np.arange(left_limit, right_limit, dt)

    # Definition of rectangle width and cosine angular velocity
    a_list = [1, 0.5, 0.1]
    w_list = [1, 8, 8*np.pi]

    # Obtaining original signals
    rec_list = []
    cs_list = []
    for i in range(len(a_list)):
        a = a_list[i]
        w = w_list[i]
        rec_list.append(rectangle_function(t, dt, -a, 2*a, 1))
        cs_list.append(np.cos(w * t))

    # Fourier Transform of original signals
    rec_FFT_list = []
    cs_FFT_list  = []
    rec_fourier_list = []
    cs_fourier_list = []
    for i in range(len(rec_list)):
        # Numerical solution
        fftRes, fftFreq = fourier_numerical((rec_list[i], cs_list[i]), dt)
        rec_FFT_list.append(abs(fftRes[0]))
        cs_FFT_list.append(abs(fftRes[1]))
        # Analytical solution
        rec_fourier_list.append((recFourier(fftFreq[0], a_list[i])))
        # missing cosine
    freq = fftFreq[0]

    # # Convolutions
    # # By numpy.convolve
    # convNP = np.convolve(rec, c, "same")
    # # By fourier transform convolution
    # conv = fftconvolve(rec, c, "same")
    #
    #
    # # Plotting of original signals
    # pl.figure("Original Signals")
    # pl.subplot(211)
    # pl.title("Rectangle")
    # pl.xlabel("time")
    # pl.ylabel("f(t)")
    # pl.plot(t,rec, label = "rec(t)")
    # pl.subplot(212)
    # pl.title("Cosine")
    # pl.xlabel("time")
    # pl.ylabel("f(t)")
    # pl.plot(t, c, label = "cos(wt)")
    #
    # # Plotting FFTs
    # pl.figure("Fourier Transforms")
    # # Rectangular Pulse Transform
    # pl.subplot(211)
    # pl.title("Rectangular Pulse Transform")
    # pl.plot(freq, recFFT / max(recFFT), label = "FFT")
    # pl.plot(freq, recFourierTransform / max(recFourierTransform), label = "Analytical Solution")
    # pl.xlim((-50, 50))
    # pl.legend()
    # # Cosine Transform
    # pl.subplot(212)
    # pl.title("Cosine Transform")
    # pl.plot(freq, cosFFT / max(cosFFT), label = "FFT")
    # pl.xlim((-w/np.pi, w/np.pi))
    # pl.legend()
    #
    # # Plotting convolutions
    # pl.figure("Convolutions")
    # pl.subplot(211)
    # pl.title("Convolution Result")
    # pl.plot(t, convNP, label = "Numpy Convolve")
    # pl.plot(t, conv, label = " FFTConvolve")
    # pl.xlabel("time")
    # pl.ylabel("f(t)")
    # pl.legend()
    # pl.subplot(212)
    # pl.title("Error")
    # pl.plot(t, abs(conv-convNP))
    # pl.xlabel("time")
    # pl.ylabel("f(t)")
    # pl.ylim((-10**(-12)), 10**(-12))
    #
    # pl.show()

