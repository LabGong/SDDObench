import numpy as np

from utils import *


def sphere(x):
    x = to2dNpArray(x)
    return np.sum(x ** 2, axis=1)


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    x = to2dNpArray(x)
    d = x.shape[1]
    part1 = -b * np.sqrt(np.sum(x ** 2, axis=1) / d)
    part2 = np.sum(np.cos(c * x), axis=1) / d
    return -a * np.exp(part1) - np.exp(part2) + a + np.e


def rosenbrock(x):
    x = to2dNpArray(x)
    x1 = x[:, :-1]
    x2 = x[:, 1:]
    return np.sum(100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2, axis=1)


def griewank(x):
    x = to2dNpArray(x)
    i = np.sqrt(np.arange(1, x.shape[1] + 1))
    return np.sum(x ** 2 / 4000, axis=1) - np.prod(np.cos(x / i), axis=1) + 1


def rastrigin(x):
    x = to2dNpArray(x)
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def mpf(x, h, w, c):
    """
    $f_1=\max_{i\in \{1,...,m\}}\frac{h_i}{1+\omega_i\sum^d_{j=1}(x_j-c_{i,j})^2}$
    """
    x = to2dNpArray(x)
    n = x.shape[0]
    x = x[:, np.newaxis, :]

    w = w[:, np.newaxis].repeat(n, axis=1).T
    w = w.reshape(w.shape[1], w.shape[2])
    h = h[:, np.newaxis].repeat(n, axis=1).T
    h = h.reshape(h.shape[1], h.shape[2])

    squared_diff = np.sum((x - c) ** 2, axis=2)
    denomi = 1 / (np.multiply(squared_diff, w) + 1)
    res = np.min(np.multiply(denomi, h), axis=1)

    return res


def get_bound(base_func):
    if base_func == "ackley":
        return -32.768, 32.768
    elif base_func == "sphere":
        return -5.12, 5.12
    elif base_func == "griewank":
        return -5, 5
    elif base_func == "rastrigin":
        return -5.12, 5.12
    elif base_func == "rosenbrock":
        return -2.048, 2.048
    elif base_func == 'mpf':
        return -5, 5
    else:
        raise Exception("no match object function.")
