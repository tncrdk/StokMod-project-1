import numpy as np
import scipy
import matplotlib.pyplot as plt

binom = scipy.stats.binom

alpha = 0.005
gamma = 0.1

def task1_e():
    N = 1000
    iterations = 300
    y = np.zeros((iterations, 3), dtype='int64')
    y[0] = np.array([950, 50, 0])

    for i in range(1, iterations):
        p_si = 0.5*y[i-1, 1] / N
        s_to_i = binom.rvs(y[i-1, 0], 0.5 * p_si)
        i_to_r = binom.rvs(y[i-1, 1], gamma)
        r_to_s = binom.rvs(y[i-1, 2], alpha)

        y[i, 0] = y[i-1, 0] - s_to_i + r_to_s
        y[i, 1] = y[i-1, 1] - i_to_r + s_to_i
        y[i, 2] = y[i-1, 2] - r_to_s + i_to_r

    plt.plot(y, label=["Susceptible", "Infected", "Recovered"])
    plt.legend()
    plt.show()

task1_e()


