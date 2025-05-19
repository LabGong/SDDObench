import numpy as np


def get_metrics(result, optimal, error_type):
    """
    result: np. Array
    optimal: np. Array
    error_type: str: "online" or "offline"
    """

    max_run, T, max_iter, d_ = result.shape

    d=d_-1

    avg_errors = np.zeros(max_run)
    online_errors = np.zeros((max_run, T, max_iter))

    for run in range(max_run):
        total_error = 0
        for t in range(T):
            cur_optimal = optimal[run, t, -1]
            if error_type == "online":
                evals = max_iter
                for iter in range(max_iter):
                    cur_evalbest = result[run, t, iter, -1]
                    cur_error = np.abs(cur_evalbest - cur_optimal)

                    total_error += cur_error
                    online_errors[run, t, iter] = cur_error
            elif error_type == "offline":
                evals = 1
                cur_evalbest = result[run, t, -1, -1]
                total_error += np.abs(cur_evalbest - cur_optimal)
            else:
                raise ValueError("error_type must be either 'online' or 'offline'.")

        avg_errors[run] = total_error / (T * evals)

    avg_errors = np.mean(avg_errors)
    online_errors = np.mean(online_errors, axis=0)

    return avg_errors, online_errors
