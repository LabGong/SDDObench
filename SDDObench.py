import numpy as np

from drift_function import *
from object_function import *
from transfromation_function import *


def sddobench(x, num_instance, df_type, change_count, dim, T, num_peaks, P=20, peak_info=None, random_seed=65,
              delta_info=None):
    """

    :param x: decision variables, a vector or an array
    :param num_instance: problem instance, int
    1: MPF with moving peaks
    2: MPF with partly changed peaks
    3: MPF with changed number of peaks
    4: DCF-sphere
    5: DCF-rosenbrock
    6: DCF-ackley
    7: DCF-griewank
    8: DCF-rastrigin

    :param df_type: drift function type, int
    1:no drift
    2:sudden drift
    3:recurrent sudden drift
    4:recurrent incremental drift
    5:recurrent incremental drift with noise

    :param change_count: times of environment change, int, start from 0
    :param dim: dimension of decision variables, int
    :param T: the maximum change times of environment, int
    :param num_peaks: peaks number of current environment, int
    :param P: the recurrent period
    :param peak_info: the environment information, list [h_peaks,w_peaks,c_peaks]
    :param random_seed:  the random seed to generate initial peaks, default is 65
    :param delta_info: the current environment information, list [change_time,delta_t]
    :return: fitness values
    """

    '''For the sake of a fair comparison, the following setting should be similar for every compared algorithm'''
    lb_h, ub_h = -70, -30  # original height
    lb_w, ub_w = 1, 12  # original width
    t_interval = 1
    t_origin = 0

    # initial setting for customized setting
    max_dim = 50
    lb_num_peaks, ub_num_peaks = 3, 40

    # parameters of transformation function
    change_rate = 0.3
    severity_h = 0.3
    severity_w = 0.1
    severity_p = 0.5
    severity_x = 0.4

    # parameters of drift function
    epsilon = 0.05 # a small magnitude of random noise
    k = 5  # the number of randomly selected time points (sudden number)
    current_peak_info = []
    t = t_origin + t_interval * change_count

    delta_t = drift_func(t, P=P, epsilon=epsilon, df_type=df_type, k=k, T=T)

    match num_instance:
        case 1 | 2 | 3:
            obf = mpf
            lb, ub = get_bound(obf.__name__)

            # # generate original peak information
            np.random.seed(random_seed)
            c_peaks_o = np.random.uniform(lb, ub, size=(ub_num_peaks, max_dim))
            h_peaks_o = np.random.uniform(lb_h, ub_h, size=(ub_num_peaks, 1))
            w_peaks_o = np.random.uniform(lb_w, ub_w, size=(ub_num_peaks, 1))
            np.random.seed(None)
            # np.savetxt('c_peaks_original.csv', c_peaks_o, delimiter=',')
            # np.savetxt('h_peaks_original.csv', h_peaks_o, delimiter=',')
            # np.savetxt('w_peaks_original.csv', w_peaks_o, delimiter=',')

            # pre-generated peak original information file
            # c_peaks_o = np.loadtxt('c_peaks_original.csv', delimiter=',')
            # h_peaks_o = np.loadtxt('h_peaks_original.csv', delimiter=',')[:,np.newaxis]
            # w_peaks_o = np.loadtxt('w_peaks_original.csv', delimiter=',')[:,np.newaxis]

            if change_count == 0:
                peak_info = [h_peaks_o[:num_peaks], w_peaks_o[:num_peaks], c_peaks_o[:num_peaks, :dim]]

            if delta_info is not None and len(peak_info) != 0 and change_count == delta_info[0]:
                # for evaluate in current environment
                h_peaks = peak_info[0]
                w_peaks = peak_info[1]
                c_peaks = peak_info[2]
            else:
                if num_instance == 1:
                    h_peaks = transform_func(h_peaks_o[:num_peaks], lb_h, ub_h, delta_t, severity=severity_h)
                    w_peaks = transform_func(w_peaks_o[:num_peaks], lb_w, ub_w, delta_t, severity=severity_w)
                    c_peaks = transform_func(c_peaks_o[:num_peaks, :dim], lb, ub, delta_t, severity=severity_x)

                    c_peaks = np.clip(c_peaks, lb, ub)
                    h_peaks = np.clip(h_peaks, lb_h, ub_h)
                    w_peaks = np.clip(w_peaks, lb_w, ub_w)

                elif num_instance == 2:
                    peaks_list = np.arange(0, num_peaks)

                    if len(peak_info) != 0:
                        h_peaks = peak_info[0]
                        w_peaks = peak_info[1]
                        c_peaks = peak_info[2]
                    else:
                        raise Exception("not defined last change peak information.")
                    changed_peak = np.random.choice(peaks_list, size=int(num_peaks * change_rate), replace=False)
                    h_peaks[changed_peak] = transform_func(h_peaks_o[changed_peak], lb_h, ub_h, delta_t, severity=severity_h)

                    w_peaks[changed_peak] = transform_func(w_peaks_o[changed_peak], lb_w, ub_w, delta_t, severity=severity_w)

                    c_peaks[changed_peak, :dim] = transform_func(c_peaks_o[changed_peak, :dim], lb, ub, delta_t,
                                                     severity=severity_x)
                    c_peaks[changed_peak] = np.clip(c_peaks[changed_peak], lb, ub)
                    w_peaks = np.clip(w_peaks, lb_w, ub_w)
                    h_peaks = np.clip(h_peaks, lb_h, ub_h)

                elif num_instance == 3:
                    if num_peaks < lb_num_peaks or num_peaks > ub_num_peaks:
                        raise Exception("Error: number of peaks are out of range.")
                    n_peaks = round(
                        transform_func(num_peaks, lb_num_peaks, ub_num_peaks, delta_t, severity=severity_p))
                    new_num_peaks = np.clip(n_peaks, lb_num_peaks, ub_num_peaks)

                    if new_num_peaks <= num_peaks:

                        h_peaks_o = h_peaks_o[:new_num_peaks]
                        w_peaks_o = w_peaks_o[:new_num_peaks]
                        c_peaks_o = c_peaks_o[:new_num_peaks]

                    else:
                        candi_peaks = np.arange(num_peaks, ub_num_peaks)
                        new_peaks = np.random.choice(candi_peaks, size=new_num_peaks - num_peaks, replace=False)
                        h_peaks_o = np.vstack((h_peaks_o[:num_peaks], h_peaks_o[new_peaks]))
                        w_peaks_o = np.vstack((w_peaks_o[:num_peaks], w_peaks_o[new_peaks]))
                        c_peaks_o = np.vstack((c_peaks_o[:num_peaks], c_peaks_o[new_peaks]))

                    h_peaks = transform_func(h_peaks_o, lb_h, ub_h, delta_t, severity=severity_h)
                    w_peaks = transform_func(w_peaks_o, lb_w, ub_w, delta_t, severity=severity_w)
                    c_peaks = transform_func(c_peaks_o[:, :dim], lb, ub, delta_t, severity=severity_x)
                    c_peaks = np.clip(c_peaks, lb, ub)
                    w_peaks = np.clip(w_peaks, lb_w, ub_w)
                    h_peaks = np.clip(h_peaks, lb_h, ub_h)

                    num_peaks = new_num_peaks

                else:
                    raise Exception("not defined MPF.")

            current_peak_info = [h_peaks, w_peaks, c_peaks]
            f = obf(x, h_peaks, w_peaks, c_peaks)

        case 4 | 5 | 6 | 7 | 8:

            if delta_info is not None and delta_info[0] == change_count:
                delta_t = delta_info[1]

            if num_instance == 4:
                obf = sphere
            elif num_instance == 5:
                obf = rosenbrock
            elif num_instance == 6:
                obf = ackley
            elif num_instance == 7:
                obf = griewank
            elif num_instance == 8:
                obf = rastrigin
            else:
                raise Exception("not defined DCF.")

            lb, ub = get_bound(obf.__name__)
            x = transform_func(x, lb, ub, delta_t, severity=severity_x)
            x = np.clip(x, lb, ub)
            f = obf(x)

        case _:
            raise Exception("no match benchmark instances")

    return f, x, current_peak_info, num_peaks, [change_count, delta_t]
