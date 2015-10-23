import numpy as np
import pylab as pl

def localAverage(grid, dimX, dimY):
    """
    This function calculates the average over the nearest points of points on a grid
    :param grid: Original points grid
    :param dimX: Grid X dimension
    :param dimY: Grid Y dimension
    :return: Grid with the local averages
    """
    # Initialization
    newGrid = np.zeros((dimX, dimY))
    lastX = dimX-1
    lastY = dimY-1
    secLastX = dimX-2
    secLastY = dimY-2

    # Average of the "inside" grid
    newGrid[1:lastX, :] += grid[2:, :] + grid[0:secLastX, :]
    newGrid[:, 1:lastY] += grid[:, 2:] + grid[:, 0:secLastY]

    # Average on the borders
    # This block puts the value of the only neighbours to the limiting rows and columns
    newGrid[:,  0]      += grid[:,  1]
    newGrid[:, lastY]   += grid[:, secLastY]
    newGrid[0, :]       += grid[1, :]
    newGrid[lastX, :]   += grid[secLastX, :]

    newGrid[0,  0]          += newGrid[1,  0] + newGrid[0,  1]
    newGrid[0, lastY]       += newGrid[1, lastY] + newGrid[0, secLastY]
    newGrid[lastX,  0]      += newGrid[secLastX,  0] + newGrid[lastX,  1]
    newGrid[lastX, lastY]   += newGrid[secLastX, lastY] + newGrid[lastX, secLastY]

    # Weighted average
    newGrid[1:lastX, 1:lastY]   *= 0.25
    newGrid[1:lastX, 0]         *= 1/3.0
    newGrid[1:lastX, lastY]     *= 1/3.0
    newGrid[0, 1:lastY]         *= 1/3.0
    newGrid[lastX, 1:lastY]     *= 1/3.0
    # Calculate the average on the corners
    newGrid[0, 0]               *= 0.5
    newGrid[0, lastY]           *= 0.5
    newGrid[lastX, 0]           *= 0.5
    newGrid[lastX, lastY]       *= 0.5

    return newGrid

def acceptancegrid(pAcceptance):
    """
    This function calculates an acceptance grid to use with the Ising model
    :param pAcceptance: grid containing the probabilities of acceptance
    :return: Grid that tells which grid elements should be replaced
    """
    randomGrid = np.random.random(pAcceptance.shape)
    return 1.0*(randomGrid < pAcceptance)

def  gridAverage(grid):
    """
    This function calculates de average value of a given grid
    :param grid: *Self explanatory*
    :return: Average value
    """
    return np.average(grid)

def gridStd(grid):
    """
    This function calculates de standard deviation of a grid
    :param grid: *Self explanatory*
    :return: The standard deviation
    """
    return np.std(grid)

if __name__  == "__main__":
    # Grid dimensions & Initial values
    iterations = 999
    dimX = 128
    dimY = 128
    gridShape = (dimX, dimY)
    N = 1
    X, Y = np.mgrid[0.0:dimX:N, 0.0:dimY:N]
    n_points = 4
    corr_points = np.linspace(1, iterations-1, n_points-1).astype(int)
    # Generation of random 2D grid with desired dim
    grid = np.random.random(gridShape)

    # Plotting of grids
    pl.figure("Original Grid and Evolution of Statistical Variables")
    # Plot of original gird
    pl.subplot(311)
    pl.title("Original Grid")
    pl.xlabel("x")
    pl.ylabel("y")
    pl.contourf(X, Y, grid, levels = np.arange(0.0, 1.1, 0.1))
    pl.colorbar()


    # Evolving the grid and statistical variables
    it = np.arange(0, iterations + 1, 1)
    av = [gridAverage(grid)]
    sigma = [gridStd(grid)]
    corr_coefs = [np.corrcoef(grid)]
    for i in range(iterations):
        # Calculation of local average
        newGrid = localAverage(grid, dimX, dimY)
        # Calculation of probabilty of acceptance
        pAcceptance = abs(grid - newGrid)
        # Calculation of acceptance gird
        acceptanceGrid = acceptancegrid(pAcceptance)
        # Generation of a evolution grid
        evolutionGrid = np.random.random(gridShape)
        # Updating the grid (Ising model)
        grid = (1 - acceptanceGrid)*grid + acceptanceGrid*evolutionGrid
        # Calculation of statistical variables
        av.append(gridAverage(grid))
        sigma.append(gridStd(grid))
        # Calculation of correlation coefficient
        if i in corr_points:
            corr_coefs.append(np.corrcoef(grid))

    # Plot of the average value evolution
    pl.subplot(312)
    pl.xlabel("iterations")
    pl.ylabel(r'$\mu$')
    pl.plot(it, av)

    # Plot of the standard deviation evolution
    pl.subplot(313)
    pl.xlabel("iterations")
    pl.ylabel(r'$\sigma$')
    pl.plot(it, sigma)

    pl.subplots_adjust(hspace = .5)

    # Plot of standard deviation close up
    pl.figure("Close Up")
    pl.title(r'$\sigma$' + ' close up')
    pl.xlabel("iterations")
    pl.ylabel(r'$\sigma$')
    pl.xlim(iterations / 2, iterations)
    pl.ylim(0.20, 0.23)
    pl.plot(it, sigma)

    # Plot of correlation coefiecients
    pl.figure("Correlation Coefficients")
    corr_points = [0] + corr_points.tolist()
    subplots_list = [221,222,223,224]
    for i in range(len(corr_coefs)):
        pl.subplot(subplots_list[i])
        pl.title("Iterarions: " + str(corr_points[i]))
        pl.xlabel("x")
        pl.ylabel("y")
        pl.contourf(X, Y, grid, levels = np.arange(0.0, 1.1, 0.1))
        pl.colorbar()

    pl.subplots_adjust(hspace = .5)


    pl.show()