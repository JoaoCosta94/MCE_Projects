import numpy as np
import pylab as pl

def systemMatrix(kArray):
    M = np.array([[kArray[0] + kArray[1], -kArray[1], 0.0],
                  [-kArray[1], kArray[1] + kArray[2], -kArray[2]],
                  [0.0, -kArray[2], kArray[2] + kArray[3]]])
    return M

def normalModes(M):
    return np.linalg.eig(A)[0]**0.5

def massMovement(X0, w):
    return X0 * np.sin(w * time)

if __name__ == '__main__':

    global time

    wArray = np.arange(0.0, 4.0, 0.1)
    dx = 0.1

    mArray = np.array([1.0, 1.0, 1.0])
    kArray = np.array([1.0, 1.0, 1.0, 1.0])
    b = np.array([0.0, 0.0, dx])

    A = systemMatrix(kArray)

    # Obtaining eigen frequencies
    nModes = normalModes(A) / mArray
    print "The eigen frequencies are:"
    print nModes

    # Obtaining normal modes for set frequency
    w = nModes[1]+ 0.2 # Chosen for graph analysis
    w2 = w*w
    W = np.array([w2,w2,w2])
    W = np.diag(W)
    x0 = np.linalg.solve(A-W,b)

    print "The normal modes are"
    print x0

    # Obtaining movements
    T = 2*np.pi / w # Oscillation period
    time = np.linspace(0.0, 3*T, 1000)
    x1 = massMovement(x0[0], w)
    x2 = massMovement(x0[1], w)
    x3 = massMovement(x0[2], w)

    # Plotting movement
    pl.figure(1)
    pl.subplot(311)
    pl.plot(time, x1)
    pl.subplot(312)
    pl.plot(time, x2)
    pl.subplot(313)
    pl.plot(time, x3)


    wArray = np.linspace(0.0, nModes[-1] + 0.5, 1000)
    A1 = []
    A2 = []
    A3 = []
    for w in wArray:
        w2 = w*w
        W = np.array([w2,w2,w2])
        W = np.diag(W)
        x0 = np.linalg.solve(A-W,b)
        A1.append(x0[0])
        A2.append(x0[1])
        A3.append(x0[2])

    A1 = np.array(A1)
    A2 = np.array(A2)
    A3 = np.array(A3)

    # Plotting spectrum
    pl.figure(2)
    pl.subplot(311)
    pl.plot(wArray, A1)
    pl.subplot(312)
    pl.plot(wArray, A2)
    pl.subplot(313)
    pl.plot(wArray, A3)

    pl.show()