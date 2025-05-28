import numpy as np

from drift_function import *
from object_function import *
from transformation_function import *
import os

curr_dir=os.path.dirname(__file__)


def sddobench(params,config):
    """
    :input
        params: dictionary containing the following keys:
            - x: decision variables, a vector or an array
            - num_instance: problem instance, int
                1: MPF with moving peaks
                2: MPF with partly changed peaks
                3: MPF with changed number of peaks
                4: DCF-sphere
                5: DCF-rosenbrock
                6: DCF-ackley
                7: DCF-griewank
                8: DCF-rastrigin
            - df_type: drift function type, int
                1:no drift
                2:sudden drift
                3:recurrent sudden drift
                4:recurrent incremental drift
                5:recurrent incremental drift with noise
            - change_count: times of environment change, int, start from 0
            - dim: dimension of decision variables, int
            - T: the maximum change times of environment, int
            - num_peaks: peaks number of current environment, int
            - P: the recurrent period, default is 20
            - peak_info: the environment information, list [h_peaks,w_peaks,c_peaks], default is None
            - random_seed: the random seed to generate initial peaks, default is 65
            - delta_info: the current environment information, dict {change_time: delta_t}, default is None

        config: dictionary for benchmark setting 
            - lb_h, ub_h: lower and upper bounds of height, default is -70, -30
            - lb_w, ub_w: lower and upper bounds of width, default is 1, 12
            - t_interval: time interval, default is 1
            - t_origin: the origin time, default is 0
            - max_dim: the maximum dimension of decision variables, default is 50
            - lb_num_peaks, ub_num_peaks: lower and upper bounds of number of peaks, default is 3, 40
            - change_rate: the rate of changed peaks, default is 0.3
            - severity_h: the severity of height change, default is 0.3
            - severity_w: the severity of width change, default is 0.1
            - severity_p: the severity of number of peaks change, default is 0.5
            - severity_x: the severity of decision variables change, default is 0.4
            - epsilon: a small magnitude of random noise, default is 0.05
            - k: the number of randomly selected time points (sudden number), default is 5
    :return: fitness values, current_peak_info, num_peaks, [change_count, delta_t]
    """
    # get the parameters from the dictionary
    x = params['x']
    num_instance = params['num_instance']
    df_type = params['df_type']
    change_count = params['change_count']
    dim = params['dim']
    T = params.get('T',60)
    num_peaks = params.get('num_peaks',8)
    P = params.get('P', 20)
    peak_info = params.get('peak_info', None)
    random_seed = params.get('random_seed', 65)
    delta_info = params.get('delta_info', None)

    '''! For the sake of a fair comparison, the following setting should be similar for every compared algorithm'''
    lb_h, ub_h = config.lb_h, config.ub_h
    lb_w, ub_w = config.lb_w, config.ub_w
    t_interval = config.t_interval
    t_origin = config.t_origin

    # initial setting for customized setting
    max_dim = config.max_dim
    lb_num_peaks, ub_num_peaks = config.lb_num_peaks, config.ub_num_peaks

    # parameters of transformation function
    change_rate = config.change_rate
    severity_h = config.severity_h
    severity_w = config.severity_w
    severity_p = config.severity_p
    severity_x = config.severity_x
    # parameters of drift function
    epsilon = config.epsilon  # a small magnitude of random noise
    k = config.k  # the number of randomly selected time points (sudden number)
    
    
    # SDDObench
    current_peak_info = []
    t = t_origin + t_interval * change_count
    delta_t = drift_func(t, P=P, epsilon=epsilon, df_type=df_type, k=k, T=T)
    obf=get_objective_func(num_instance)
    lb,ub=get_bound(num_instance)
    match num_instance:
        case 1 | 2 | 3:
            # # generate original peak information (if user redefine the peak information, this part should be excuted)
            # c_peaks_o_path=os.path.join(curr_dir, 'peak_initial/c_peaks_original_new.csv')
            # h_peaks_o_path=os.path.join(curr_dir, 'peak_initial/h_peaks_original_new.csv')
            # w_peaks_o_path=os.path.join(curr_dir, 'peak_initial/w_peaks_original_new.csv')
            # np.random.seed(random_seed)
            # c_peaks_o = np.random.uniform(lb, ub, size=(ub_num_peaks, max_dim))
            # h_peaks_o = np.random.uniform(lb_h, ub_h, size=(ub_num_peaks, 1))
            # w_peaks_o = np.random.uniform(lb_w, ub_w, size=(ub_num_peaks, 1))
            # np.random.seed(None)
            # np.savetxt(c_peaks_o_path, c_peaks_o, delimiter=',')
            # np.savetxt(h_peaks_o_path, h_peaks_o, delimiter=',')
            # np.savetxt(w_peaks_o_path, w_peaks_o, delimiter=',')

            # pre-generated peak original information file (defult setting)
            c_peaks_o_path=os.path.join(curr_dir, 'peak_initial/c_peaks_original.csv')
            h_peaks_o_path=os.path.join(curr_dir, 'peak_initial/h_peaks_original.csv')
            w_peaks_o_path=os.path.join(curr_dir, 'peak_initial/w_peaks_original.csv')
            c_peaks_o = np.loadtxt(c_peaks_o_path, delimiter=',')
            h_peaks_o = np.loadtxt(h_peaks_o_path, delimiter=',')[:,np.newaxis]
            w_peaks_o = np.loadtxt(w_peaks_o_path, delimiter=',')[:,np.newaxis]

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
                    h_peaks[changed_peak] = transform_func(h_peaks_o[changed_peak], lb_h, ub_h, delta_t,
                                                           severity=severity_h)

                    w_peaks[changed_peak] = transform_func(w_peaks_o[changed_peak], lb_w, ub_w, delta_t,
                                                           severity=severity_w)

                    c_peaks[changed_peak, :dim] = transform_func(c_peaks_o[changed_peak, :dim], lb, ub, delta_t,
                                                                 severity=severity_x)
                    c_peaks[changed_peak] = np.clip(c_peaks[changed_peak], lb, ub)
                    w_peaks = np.clip(w_peaks, lb_w, ub_w)
                    h_peaks = np.clip(h_peaks, lb_h, ub_h)

                elif num_instance == 3:
                    if num_peaks < lb_num_peaks or num_peaks > ub_num_peaks:
                        raise Exception("Error: number of peaks are out of range.")
                    n_peaks = round(transform_func(num_peaks, lb_num_peaks, ub_num_peaks, delta_t, severity=severity_p).item())
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
            y = obf(x, h_peaks, w_peaks, c_peaks)
            params.update(peak_info=current_peak_info, num_peaks=num_peaks)

        case 4 | 5 | 6 | 7 | 8:

            if delta_info is not None and delta_info[0] == change_count: # define the drift intformation in the current environment
                delta_t = delta_info[1]
            x = transform_func(x, lb, ub, delta_t, severity=severity_x)
            x = np.clip(x, lb, ub)
            y = obf(x)

        case _:
            raise Exception("no match benchmark instances")
        
    params.update(delta_info=[change_count, delta_t])
    return y, params

