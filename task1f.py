import numpy as np
import scipy
import matplotlib.pyplot as plt

binom = scipy.stats.binom  # General binomial distribution
t = scipy.stats.t  # General student-t distribution

# Constants
alpha = 0.005
gamma = 0.1


def task1_e():
    N = 1000
    iterations = 300
    inf_max_arr = np.zeros(N)
    inf_max_time_arr = np.zeros(N)

    for k in range(N):
        y = np.zeros((iterations, 3), dtype='int64')
        y[0] = np.array([950, 50, 0])

        for i in range(1, iterations):
            p_si = 0.5 * y[i - 1, 1] / N
            s_to_i = binom.rvs(y[i - 1, 0], p_si)
            i_to_r = binom.rvs(y[i - 1, 1], gamma)
            r_to_s = binom.rvs(y[i - 1, 2], alpha)

            y[i, 0] = y[i - 1, 0] - s_to_i + r_to_s
            y[i, 1] = y[i - 1, 1] - i_to_r + s_to_i
            y[i, 2] = y[i - 1, 2] - r_to_s + i_to_r

        # Find the max value and time of the first occurence of the max value
        inf_max_arr[k] = np.max(y[:, 1])
        inf_max_time_arr[k] = np.min(np.argmax(y[:, 1]))

    # Find the estimates for max value and time of first occurence
    est_inf_max = np.mean(inf_max_arr)
    est_inf_max_time = np.mean(inf_max_time_arr)

    # calculating sample standard deviation
    s_max = np.std(inf_max_arr, ddof=1)
    s_time = np.std(inf_max_time_arr, ddof=1)

    # the 0.975 percentile to be used late, this can be used for lower and upper, since t-dist is symmetric
    t_alpha = t.ppf(0.975, N - 1)

    # Finding the quantiles
    lower_inf_max = est_inf_max - t_alpha * s_max / np.sqrt(N)
    upper_inf_max = est_inf_max + t_alpha * s_max / np.sqrt(N)

    lower_inf_max_time = est_inf_max_time - t_alpha * s_time / np.sqrt(N)
    upper_inf_max_time = est_inf_max_time + t_alpha * s_time / np.sqrt(N)

    print(f"estimate infected max {est_inf_max}")
    print(f"estimate infected max time {est_inf_max_time}")

    print(f"interval max {lower_inf_max}, {upper_inf_max}")
    print(f"interval time {lower_inf_max_time}, {upper_inf_max_time}")


task1_e()
