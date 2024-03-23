import numpy as np

def calculate_metric(result,optimal,perf_type):
    """
    Args:
        result: the result obtained by algorithms, an array (max_run,T,max_item,dim)
        optimal: the optimum, an array (max_run,T,dim)
        perf_type: performance type
        1--> online error
        2--> offline error

    Returns:
        performance metric
    """
    max_item=30 # the iterate generation in every environment
    T,max_run=result.shape

    error_s=np.zeros((max_run))

    for run in range(max_run):
        cureval_error = 0
        curtrue_optimal=optimal[run,t]
        for t in range(T):
            if perf_type==1:
                evals=max_item
                for item in range(max_item):
                    curevalbest=result[run,t,item]
                    cureval_error+=np.abs(curtrue_optimal-curevalbest)
            elif perf_type==2:
                evals=1
                curevalbest=result[run,t,-1]
                cureval_error += np.abs(curtrue_optimal - curevalbest)
            else:
                raise Exception("no defined performance type.")
        error_s[run]=cureval_error/(T*evals)

    return error_s

