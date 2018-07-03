from scipy.spatial import distance
from scipy.linalg import norm
import numpy as np


def calc_distance(x, y):
    return distance.euclidean(x, y)


def calc_norm(x):
    return norm(x)


def calc_func_gradient(func, x, f0):
    """
    Gradient func with respect to x
    :param func: constraint function
    :param x: point at which gradient is calculated
    :param f0: constraint evaluated at x
    :return: gradient
    """
    n_dim = np.size(x)
    dx_i = 0.001
    grad_vec = np.empty(n_dim, dtype="float32")

    for i in range(n_dim):
        sign = 1
        if x[i]+dx_i >= 1:  # since hypercube
            sign = -1       # backward difference
        x[i] += sign*dx_i
        grad_vec[i] = sign*(eval(func)-f0)/dx_i
        x[i] -= sign*dx_i

    return grad_vec


def calc_dx(func, x, fx):
    """
    Trying to get dx for constraints to reach 0, if current constraint value is negative
    Solves Linear undetermined system, df = H dx. Minimizes ||dx||. df = 0-fx, H is gradient (row vector)
    Return dx = H'(HH')^(-1) df
    :param func: constraint function
    :param x: point at which gradient is calculated
    :param fx: constraint evaluated at x
    :return: dx to satisfy constraint
    """
    n_dim = np.size(x)
    if fx >= 0:
        return np.zeros(n_dim, dtype="float32")
    else:
        df = 0-fx
        gradient = calc_func_gradient(func, x, fx)      # O(N^2)
        rhs = 1.0/(np.dot(gradient, gradient))*df       # O(N)
        dx = gradient*rhs

        return dx


    # average distances from dx from all constraints?
    # can't just add distances, since going to normalize using branch's max distance
    # note, doing this way (one by one), we are under estimating actual dx

    # combine vectors, not as risk of running out of bounds like distances
    # can't just add, because dx, for one could be lower than another
    # if all positive or negative, easier
    # max of all dimensions, that way make up for underestimating
    # but if opposite, trickier, using max(abs), only need distance anyway