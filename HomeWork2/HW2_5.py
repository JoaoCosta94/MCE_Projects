import numpy as np
import pylab as pl
import HW2_4 as hw4

def systemMatrix(kArray):
    """
    This function creates the matrix of a system with 3 masses and 4 springs
    :param kArray:  Array with spring constant's
    :return:        K matrix
    """
    M = np.array([[kArray[0] + kArray[1], -kArray[1], 0.0],
                  [-kArray[1], kArray[1] + kArray[2], -kArray[2]],
                  [0.0, -kArray[2], kArray[2] + kArray[3]]])
    return M

def normalModes(K):
    """
    This function calculates the eigen frequencies of a mass-spring system
    :param K:   K matrix of the system
    :return:    Array with the eigen frequencies
    """
    return np.linalg.eig(K)[0]**0.5

def massMovement(X0, w):
    """
    This function calculates the oscillation of each mass on the mass-spring system
    :param X0:      Amplitude of the movement
    :param w:       Oscillation frequency
    :return:        Array with mass positions over time
    """
    return X0 * np.sin(w * time)

if __name__ == '__main__':

    global time
    # Initialization
    mArray = np.array([1.0, 1.0, 1.0]) # masses
    kArray = np.array([1.0, 1.0, 1.0, 1.0]) # spring constants
    dx = 0.1 # amplitude of external source's movement

    # this is the array b for the Ax = b equation
    b = np.array([0.0, 0.0, dx])

    # this is the matrix A for the Ax = b equation corresponds to the K matrix of a mass-spring system
    A = systemMatrix(kArray)

    # Obtaining eigen frequencies of the mass-spring system : det(K-W/m) = 0
    nModes = normalModes(A) / mArray
    print "The eigen frequencies are:"
    print nModes

    # Obtaining normal modes for set frequency
    w = nModes[1] # Chosen for graph analysis
    w2 = w*w
    W = np.array([w2,w2,w2])
    W = np.diag(W) # Diagonal matrix with the operation frequency
    x0 = hw4.gaussElimination(A-W, b)

    print "The normal modes are"
    print x0

    # Obtaining movements
    T = 2*np.pi / w # Oscillation period
    time = np.linspace(0.0, 3*T, 1000) # Movement in 3 periods
    # Movement for each mass
    x1 = massMovement(x0[0], w)
    x2 = massMovement(x0[1], w)
    x3 = massMovement(x0[2], w)

    # Obtaining spectrum
    wArray = np.linspace(0.0, np.max(nModes) + 0.5, 1000)
    A1 = []
    A2 = []
    A3 = []
    for w in wArray:
        w2 = w*w
        W = np.array([w2,w2,w2])
        W = np.diag(W)
        x0 = hw4.gaussElimination(A-W, b)
        # Amplitude for each mass
        A1.append(x0[0])
        A2.append(x0[1])
        A3.append(x0[2])

    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)

    # Plotting movement
    pl.figure("Mass movement")
    pl.subplot(311)
    pl.title("Mass 1 oscillation")
    pl.xlabel("t")
    pl.ylabel("X1")
    pl.xlim(0.0, 3*T)
    pl.plot(time, x1)

    pl.subplot(312)
    pl.title("Mass 2 oscillation")
    pl.xlabel("t")
    pl.ylabel("X1")
    pl.xlim(0.0, 3*T)
    pl.plot(time, x2)

    pl.subplot(313)
    pl.title("Mass 3 oscillation")
    pl.xlabel("t")
    pl.ylabel("X1")
    pl.xlim(0.0, 3*T)
    pl.plot(time, x3)

    pl.subplots_adjust(hspace = 0.7)

    # Plotting spectrum
    # TODO: Add vertical lines at eigen frequencies
    xMin = 0.0
    xMax = np.max(nModes) + 0.5
    pl.figure("Spectrum")

    pl.subplot(311)
    pl.title("X1 frequency response")
    pl.xlim(xMin, xMax)
    pl.xlabel(r'$\omega$')
    pl.ylabel("X1")
    pl.plot(wArray, A1)

    pl.subplot(312)
    pl.title("X2 frequency response")
    pl.xlim(xMin, xMax)
    pl.xlabel(r'$\omega$')
    pl.ylabel("X2")
    pl.plot(wArray, A2)
    pl.subplot(313)

    pl.title("X3 frequency response")
    pl.xlim(xMin, xMax)
    pl.xlabel(r'$\omega$')
    pl.ylabel("X3")
    pl.plot(wArray, A3)

    pl.subplots_adjust(hspace = 0.7)

    pl.show()