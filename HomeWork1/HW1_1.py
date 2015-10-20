import pylab as pl
import numpy as np
from scipy.signal import fftconvolve
from time import time

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

def cosFourier(f, w):
    # TODO: Finish this
    # 3 points around desired frequency
    w = w / (2.0 * np.numpy)
    return "cenas"


def plotFTRectangluar(rec_FFT_list, rec_fourier_list, fftFreq):
    pl.figure("Fourier Transforms - Rectangular Pulses")
    pl.subplot(211)
    pl.title("FFT")
    pl.xlabel("f")
    pl.ylabel("F(f)")
    for i in range(len(rec_list)):
        pl.plot(fftFreq, rec_FFT_list[i] / max(rec_FFT_list[i]), label = "FFT - a = " + str(a_list[i]))
    pl.legend()
    pl.subplot(212)
    pl.title("Analytical FT")
    pl.xlabel("f")
    pl.ylabel("F(f)")
    for i in range(len(rec_list)):
        pl.plot(fftFreq, rec_fourier_list[i] / max(rec_fourier_list[i]), label = "Analytical - a = " + str(a_list[i]))
    pl.legend()
    # pl.subplot(313)
    # pl.title("Difference")
    # pl.xlabel("f")
    # pl.ylabel("Error")
    # for i in range(len(rec_list)):
    #     pl.plot(fftFreq, abs(rec_fourier_list[i] / max(rec_fourier_list[i]) - rec_FFT_list[i] / max(rec_FFT_list[i])),
    #             label = "Error - a = " + str(a_list[i]))
    # pl.legend()

def plotFTCosine(cs_FFT_list, cs_fourier_list, fftFreq):
    pl.figure("Fourier Transforms - Cosines")
    pl.subplot(111)
    pl.title("FFT")
    pl.xlabel("f")
    pl.ylabel("F(f)")
    for i in range(len(cs_list)):
        pl.plot(fftFreq, cs_FFT_list[i] / max(cs_FFT_list[i]), label = "w = " + str(w_list[i]))
    pl.legend()
    # pl.subplot(312)
    # pl.title("Analytical FT")
    # pl.xlabel("f")
    # pl.ylabel("F(f)")
    # for i in range(len(cs_list)):
    #     pl.plot(fftFreq, cs_fourier_list[i] / max(cs_fourier_list[i]), label = "Analytical - w = " + str(w_list[i]))
    # pl.legend()
    # pl.subplot(313)
    # pl.title("Difference")
    # pl.xlabel("f")
    # pl.ylabel("Error")
    # for i in range(len(cs_list)):
    #     pl.plot(fftFreq, abs(cs_fourier_list[i] / max(cs_fourier_list[i]) - cs_FFT_list[i] / max(cs_FFT_list[i])),
    #             label = "Error - a = " + str(a_list[i]))
    # pl.legend()

def plotConvolutios(conv_NP_list, conv_list):
    pl.figure("Convolutions")
    pl.subplot(311)
    pl.title("a = " + str(a_list[0]) +" w = " + str(w_list[0]))
    pl.xlabel("t")
    pl.ylabel ("f X g")
    pl.plot(t, conv_NP_list[0], label = "NP Convolve")
    pl.plot(t, conv_list[0], label = "FFT Convolve")

    pl.subplot(312)
    pl.title("a = " + str(a_list[1]) +" w = " + str(w_list[1]))
    pl.xlabel("t")
    pl.ylabel ("f X g")
    pl.plot(t, conv_NP_list[1], label = "NP Convolve")
    pl.plot(t, conv_list[1], label = "FFT Convolve")

    pl.subplot(313)
    pl.title("a = " + str(a_list[2]) +" w = " + str(w_list[2]))
    pl.xlabel("t")
    pl.ylabel ("f X g")
    pl.plot(t, conv_NP_list[2], label = "NP Convolve")
    pl.plot(t, conv_list[2], label = "FFT Convolve")

if __name__  == "__main__":

    # Auxiliary variables to plotting
    global colors
    global linestyles
    colors = {0: 'b', 1:'g', 2:'r', 3:'y'}
    linestyles = {0: '-', 1:'--', 2:':', 3:'-.'}

    # Definition of general conditions of margins and time (may be changed)
    left_limit = -10
    right_limit = 10
    dt = 0.01
    global t
    t = np.arange(left_limit, right_limit, dt)

    # Definition of rectangle width and cosine angular velocity (may be changed)
    global a_list
    global w_list
    a_list = [1, 0.5, 0.1]
    w_list = [1, 8, 8*np.pi]

    # Obtaining original signals
    rec_list = []
    cs_list = []
    start = time()
    for i in range(len(a_list)):
        a = a_list[i]
        w = w_list[i]
        rec_list.append(rectangle_function(t, dt, -a, 2*a, 1))
        cs_list.append(np.cos(w * t))
    end = time()
    print "Calculating original signals took " + str (end-start)

    # Plotting original signals
    pl.figure("Original Signals")
    for i in range(len(rec_list)):
        pl.subplot(211)
        pl.title("Rectangular Pulses")
        pl.xlim((-left_limit, left_limit))
        pl.ylim((0, max(a_list)+0.15))
        pl.plot(t, rec_list[i], linestyle = linestyles[i], label = "a = " + str(a_list[i]))
        pl.subplot(212)
        pl.title("Cosines")
        pl.xlim((-left_limit, left_limit))
        pl.ylim((-1.2, 1.2))
        pl.plot(t, cs_list[i],label = "w = " + str(w_list[i]))
    pl.legend()

    # Fourier Transform of original signals
    rec_FFT_list = []
    cs_FFT_list  = []
    rec_fourier_list = []
    cs_fourier_list = []
    start = time()
    for i in range(len(rec_list)):
        # Numerical solutions
        fftRes, fftFreq = fourier_numerical((rec_list[i], cs_list[i]), dt)
        rec_FFT_list.append(abs(fftRes[0]))
        cs_FFT_list.append(abs(fftRes[1]))
        # Analytical solutions
        rec_fourier_list.append(abs((recFourier(fftFreq[0], a_list[i]))))
        cs_fourier_list.append(abs(cosFourier(fftFreq[0], w_list[i])))

    # Plotting of Fourier Transforms
    plotFTRectangluar(rec_FFT_list, rec_fourier_list, fftFreq[0])
    plotFTCosine(cs_FFT_list, cs_fourier_list, fftFreq[0])

    # Convolutions
    # By numpy.convolve
    conv_NP_list =[]
    conv_list = []
    times_NP = []
    times_FFT = []
    for i in range(len(rec_list)):
        start = time()
        conv_NP_list.append(np.convolve(rec_list[i], cs_list[i], "same"))
        end = time()
        times_NP.append(end-start)
        conv_list.append(fftconvolve(rec_list[i], cs_list[i], "same"))
        end_2 = time()
        times_FFT.append(end_2-end)

    # Plotting Convolutions
    plotConvolutios(conv_NP_list, conv_list)
    pl.show()

