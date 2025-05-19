from SDDObench import *
import numpy as np
from tqdm import tqdm

insts = np.arange(1, 9)
drfs = np.arange(1, 6)
samples = 20
dim = 2
T = 60
P = 20
num_peaks = 8

if __name__ == "__main__":
    for instance in insts:
        for dfind in drfs:
            lb, ub = get_bound(instance)
            x_samp = np.linspace(lb, ub, samples)
            x0, x1 = np.meshgrid(x_samp,x_samp)
            X = np.column_stack((x0.ravel(), x1.ravel()))
            params={'num_instance':instance,
                    'df_type':dfind,
                    'num_peaks': num_peaks, 
                    'T': T, 
                    'P': P,
                    'dim':dim,
                    'peak_info': None, 
                    'delta_info': None}
            for t in tqdm(range(T),desc=f'F:{instance}/8 D:{dfind}/5'):
                params.update(x=X,change_count=t)
                y,params= sddobench(params)
