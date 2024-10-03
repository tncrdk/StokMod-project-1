import numpy as np
import scipy
import matplotlib.pyplot as plt

binom = scipy.stats.binom

alpha = 0.005
gamma = 0.1


def simulation(
    time_steps: int, y0: np.ndarray, n_vaccinated: int, alpha: float, gamma: float
) -> np.ndarray:
    """Simulate one realization of the markov chain."""
    y = np.zeros((3, time_steps), dtype=np.int64)  # S = 0, I = 1, R = 2
    y[:, 0] = y0
    N = np.sum(y0)  # Population size

    for i in range(1, time_steps):
        p_si = 0.5 * y[1, i - 1] / N
        n_si = binom.rvs(y[0, i - 1], p_si)  # Number of people becoming infected
        n_ir = binom.rvs(y[1, i - 1], gamma)  # Number of people recovering
        # Subtract the number of vaccinated, as they are permanently recovered
        # and should not be concidered becoming susceptible
        n_rs = binom.rvs(
            (y[2, i - 1] - n_vaccinated), alpha
        )  # Number of people getting susceptible.

        # Update the flow of people
        y[0, i] = y[0, i - 1] + n_rs - n_si
        y[1, i] = y[1, i - 1] + n_si - n_ir
        y[2, i] = y[2, i - 1] + n_ir - n_rs

    return y


def calculate_CI(n_simulations: int, time_steps: int, y0: np.ndarray, n_vaccinated: int):
    max_infected_arr = np.zeros(n_simulations)
    max_infected_time_arr = np.zeros(n_simulations)
    alpha = 0.005
    gamma = 0.1

    for i in range(n_simulations):
        y = simulation(time_steps, y0, n_vaccinated, alpha, gamma)
        max_infected_index = np.argmax(y[1, :])
        max_infected_time_arr[i] = max_infected_index
        max_infected_arr[i] = y[1, max_infected_index]

    avg_max_infected = np.mean(max_infected_arr)
    std_max_infected = np.std(max_infected_arr)

    avg_max_infected_time = np.mean(max_infected_time_arr)
    std_max_infected_time = np.std(max_infected_time_arr)
    
    lower_max_infected = 



def task1_g():
    n_simulations = 1000
    time_steps = 300


task1_g()
