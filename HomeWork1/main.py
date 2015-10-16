import pylab as pl

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
#   generates a array with nPoints zeros
    rec     = pl.zeros(x.shape)
#   generates an array of h value with desired width
    heights = h * pl.ones((abs(w-left_limit) / spacing,))

#
    index_left = abs(int(left_limit / spacing))
    index_right = int(w / spacing) - index_left
§§left:index_right + 1] = heights
§§

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

def e1():

#   creation of original functions
    x = pl.arange(-2, 2, 0.01)
    rec = rectangle_function(x, 0.01, -1, 2, 1)
    pl.plot(x, pl.cos(6*x))
    pl.plot(x, rec)

e1()
pl.show()