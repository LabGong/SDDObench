import numpy as np


def drift_func(t, P, epsilon, df_type, k, T):
    match df_type:
        case 1:
            res = no_df()
        case 2:
            res = sudden_df(t=t, k=k, T=T)
        case 3:
            res = re_sudden_df(t=t, P=P)
        case 4:
            res = re_incremental_df(t=t, P=P)
        case 5:
            res = re_incremental_noise_df(t=t, P=P, epsilon=epsilon)
        case _:
            raise Exception("Error: no match drift function type.")
    return res


def no_df():
    return 0


def sudden_df(t, k=8, T=8):
    if t == 0:
        return 0
    flag = False
    ts = 0
    np.random.seed(int(k))
    big_lamda = np.sort(np.random.choice(np.arange(1, T + 1), size=k, replace=False))
    t = int(t)
    for i in big_lamda:
        if t == i:
            flag = True
        elif t > i:
            ts = i
    if flag:
        np.random.seed(t)
    else:
        np.random.seed(ts)
    res = np.random.uniform(-1, 1)
    np.random.seed(None)
    return res


def re_sudden_df(t, P=20):
    if t == 0:
        return 0
    t %= P
    t_c = int(0.5 * P)
    if t < t_c:
        res = -0.5
    else:
        res = 0.5
    return res


def re_incremental_df(t, P=20):
    if t % P == 0:
        return 0
    res = np.cos(2 * np.pi * t / P + np.pi / 2)
    return res


def re_incremental_noise_df(t, P=20, epsilon=0.01):
    if t % P == 0:
        return 0
    noise = epsilon * np.random.uniform(-1, 1)
    res = (1 - epsilon) * np.cos(2 * np.pi * t / P + np.pi / 2) + noise
    return res
