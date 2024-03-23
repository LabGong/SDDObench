from SDDObench import *
import numpy as np

ins = np.arange(1, 9)
drfs = np.arange(1, 6)
obj_list = ['', 'mpf', 'mpf', 'mpf', 'sphere', 'rosenbrock', 'ackley', 'griewank', 'rastrigin']
samples = 20
dim = 2
T = 200
P = 20
num_peaks = 8

if __name__ == "__main__":
    for instance in ins:
        for dfind in drfs:
            lb, ub = get_bound(obj_list[instance])
            x_samp = np.linspace(lb, ub, samples)
            x0, x1 = np.meshgrid(x_samp,x_samp)
            X = np.column_stack((x0.ravel(), x1.ravel()))

            peaks_info = []
            global_pos = np.empty([T, dim])
            for t in range(T):
                y, _, peaks_info, num_peaks, delta_info = sddobench(X, num_instance=instance, df_type=dfind,
                                                                    change_count=t, dim=dim, T=T, num_peaks=num_peaks,
                                                                    P=P, peak_info=peaks_info)
