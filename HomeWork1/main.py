import pylab as pl
import numpy as np

def rectangle_function(x, spacing, left_limit, w, h):
    """
    This function generates a rectangle with user's desired properties
    :param nPoints:     Number of points on resulting array
    :param spacing:     Spacing between consecutive points
    :param left_limit:  Index where the rectangle begins
    :param w:           Rectangle width
    :param h:           Height of the rectangle
    :return:            Array with desired rectangle
    """
    # generates a array with nPoints zeros
    rec = pl.zeros(x.shape)

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
    heights = h*pl.ones((w / spacing + 1,))

    # rearranging the rectangle
    rec[index_left:index_right+1] = heights
    return rec

def cosine(x, left_limit, right_limit, spacing, w):
    """
    This function generates a cosine with user's desired properties
    :param nPoints:     Number of points on resulting array
    :param spacing:     Spacing between consecutive points
    :param w:           Angular velocity
    :return:            Array with desired cosine function
    """
    x = pl.arange(left_limit, right_limit, spacing)
    return pl.cos(w * x)

def fourier_numerical(signalsList):

    """
    This method calculates given signals FFT using numpy.fft
    :param signalsList: List with signals array to transform
    :return: Tuple with list of FFT of original signals and list of FFT frequencies
    """

    signalsFFT = []
    fftFreqs = []
    for signal in signalsList:
        signalsFFT.append(np.fft.fft(signal))
        fftFreqs.append(np.fft.fftfreq(signal.size))

    return (signalsFFT, fftFreqs)

# Definition of general conditions of margins and time (may be changed)
left_limit = -5
right_limit = 5
dt = 0.01
t = pl.arange(left_limit, right_limit, dt)

# Definition of rectangle
a = 2 # This may be changed
rec = rectangle_function(t, dt, -a, 2*a, 1)
# Definition of original cosine function
w = 5*2*np.pi # This may be changed
c = pl.cos(w * t)

# FFT of original signals
fft, fftFreq = fourier_numerical((rec, c))
recFFT = fft[0]
cFFT = fft[1]

# Convolutions
# By numpy.convolve
convNP = np.convolve(rec, c, "same")
# By convolution theorem
conv = recFFT * cFFT
conv = np.fft.ifft(conv)
print len(conv)


# Plotting of original signals
pl.figure(1)
pl.title("Original Signals")
pl.xlabel("time")
pl.ylabel("f(t)")
pl.plot(t,rec, label = "rec(t)")
pl.plot(t, c, label = "cos(wt)")
pl.legend()

# Plotting FFTs
pl.figure(2)
pl.title("REC FFT")
pl.plot(fftFreq[0], recFFT.real)
pl.figure(3)
pl.title("COS FFT")
pl.plot(fftFreq[1], cFFT.real)

# Plotting convolutions
pl.figure(4)
pl.title("Convolution by Numpy Convolve")
pl.plot(t, convNP)
pl.figure(5)
pl.title("Convolution by Convolution Theorem")
pl.plot(t, conv)


pl.show()

