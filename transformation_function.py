import numpy as np


def transform_func(phi, lb, ub, delt_t, severity):
    delta = ub - lb
    if delta == 0:
        raise Exception("Error: lower bound equal to upper bound.")
    if isinstance(phi,np.ndarray):
        shift_vec = np.full(phi.shape, delt_t)
        dim = phi.shape[0] if len(phi.shape) == 1 else phi.shape[1]
    else:
        shift_vec = delt_t
        dim=1

    theta = 4 * np.arcsin(delt_t ** 2)

    lamda = severity * delta

    rotate_mart = get_rotate_mart(theta, dim)

    shift_res = phi + lamda * shift_vec

    if dim == 1:
        return shift_res * rotate_mart
    else:
        return shift_res @ rotate_mart


def get_rotate_mart(theta, dim):
    n = (dim - 1) * (dim % 2) + dim * (1 - (dim % 2))

    l = np.arange(dim)
    l = l[:n]
    rotate_mart = np.eye(dim)
    for i in range(int(n / 2)):
        rotate_mart[l[i * 2], l[i * 2]] = np.cos(theta)
        rotate_mart[l[i * 2 + 1], l[i * 2 + 1]] = np.cos(theta)
        rotate_mart[l[i * 2], l[i * 2 + 1]] = -np.sin(theta)
        rotate_mart[l[i * 2 + 1], l[i * 2]] = np.sin(theta)
    return rotate_mart
