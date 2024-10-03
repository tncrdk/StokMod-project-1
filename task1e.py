import numpy as np
import scipy
import matplotlib.pyplot as plt

binom = scipy.stats.binom  # A general binomial distribution

# Constants
alpha = 0.005
gamma = 0.1


def task1_e():
    N = 1000
    iterations = 300
    y = np.zeros((iterations, 3), dtype='int64')  # Where all the states are stored
    y[0] = np.array([950, 50, 0])  # Starting state of the Markov chain

    # All the iterations except 0
    for i in range(1, iterations):
        p_si = 0.5 * y[i - 1, 1] / N  # Calculate new beta_n to be used later
        # Use binomial since there are only 2 outcomes with probabilities that are known
        s_to_i = binom.rvs(y[i - 1, 0], p_si)
        i_to_r = binom.rvs(y[i - 1, 1], gamma)
        r_to_s = binom.rvs(y[i - 1, 2], alpha)

        # Update the next values
        y[i, 0] = y[i - 1, 0] - s_to_i + r_to_s
        y[i, 1] = y[i - 1, 1] - i_to_r + s_to_i
        y[i, 2] = y[i - 1, 2] - r_to_s + i_to_r

    plt.plot(y, label=["Susceptible", "Infected", "Recovered"])
    plt.legend()
    plt.title(rf"Temporal evolution of one realization of $Y_n$")
    plt.ylabel("Individuals")
    plt.xlabel("time step")
    plt.savefig("plots/task1e.pdf")


task1_e()
