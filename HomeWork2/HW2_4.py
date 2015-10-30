import numpy as np

def luDecomp(M):
    return np.linalg.lu(M)

if __name__ == '__main__':

    # Initialization
    global A
    global b
    global x0

    