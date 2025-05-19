import numpy as np



def sphere(x):
    x = np.atleast_2d(x)
    return np.sum(x ** 2, axis=1)


def ackley(x, a=20, b=0.2, c=2 * np.pi):
    x = np.atleast_2d(x)
    d = x.shape[1]
    part1 = -b * np.sqrt(np.sum(x ** 2, axis=1) / d)
    part2 = np.sum(np.cos(c * x), axis=1) / d
    return -a * np.exp(part1) - np.exp(part2) + a + np.e


def rosenbrock(x):
    x = np.atleast_2d(x)
    x1 = x[:, :-1]
    x2 = x[:, 1:]
    return np.sum(100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2, axis=1)


def griewank(x):
    x = np.atleast_2d(x)
    i = np.sqrt(np.arange(1, x.shape[1] + 1))
    return np.sum(x ** 2 / 4000, axis=1) - np.prod(np.cos(x / i), axis=1) + 1


def rastrigin(x):
    x = np.atleast_2d(x)
    return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x), axis=1)


def mpf(x, h, w, c):
    r'''
    $f_1=\min_{i\in \{1,...,m\}}\frac{h_i}{1+\omega_i\sum^d_{j=1}(x_j-c_{i,j})^2}$
    '''
    x = np.atleast_2d(x)
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



def get_objective_func(num_instance):
    if num_instance in [1,2,3]:
        return mpf
    elif num_instance==4:
        return sphere
    elif num_instance==5:
        return rosenbrock
    elif num_instance==6:
        return ackley
    elif num_instance==7:
        return griewank
    elif num_instance==8:
        return rastrigin
    else:   
        raise Exception(f"F should be range[1,8], but got F{num_instance} ")
    
def get_bound(num_instance):
    if num_instance in [1,2,3]:
        return -5, 5
    elif num_instance==4:
        return -5.12, 5.12
    elif num_instance==5:
        return -2.048, 2.048
    elif num_instance == 6:
        return -32.768, 32.768
    elif num_instance==7:
        return -5, 5
    elif num_instance==8:
        return -5.12, 5.12
    else:
        raise Exception(f"F should be range[1,8], but got F{num_instance}")