import numpy as np
import scipy
import matplotlib.pyplot as plt

binom = scipy.stats.binom
t = scipy.stats.t

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

def plot_realization(time_steps: int, y0: np.ndarray, n_vaccinated: int):
    alpha = 0.005
    gamma = 0.1

    t = np.linspace(0, time_steps, time_steps)
    y = simulation(time_steps, y0, n_vaccinated, alpha, gamma)

    plt.plot(t, y.T, label=["Susceptible", "Infected", "Recovered"])
    plt.title("Temporal Evolution of one realization of Y")
    plt.xlabel("Time Step [days]")
    plt.ylabel("Individuals")
    plt.legend()
    plt.savefig(f"plots/realization-{n_vaccinated}.pdf")
    plt.clf()


def calculate_CI(
    n_simulations: int, time_steps: int, y0: np.ndarray, n_vaccinated: int
):
    # Create arrays for containing necessary data
    max_infected_arr = np.zeros(n_simulations)
    max_infected_time_arr = np.zeros(n_simulations)

    # Initialize params
    alpha = 0.005
    gamma = 0.1

    for i in range(n_simulations):
        y = simulation(time_steps, y0, n_vaccinated, alpha, gamma)
        # Find index of maximum
        max_infected_index = np.argmax(y[1, :])
        # Save the maximum value and index(time) of the value.
        max_infected_time_arr[i] = max_infected_index
        max_infected_arr[i] = y[1, max_infected_index]

    # Take averages
    avg_max_infected = np.mean(max_infected_arr)
    std_max_infected = np.std(max_infected_arr, ddof=1)

    # Empirical standard deviation
    avg_max_infected_time = np.mean(max_infected_time_arr)
    std_max_infected_time = np.std(max_infected_time_arr, ddof=1)

    # We assume the average is approximatley normally distributed, so our estimator
    # is Student-t distributed
    t_alpha = t.ppf(0.975, n_simulations - 1)

    # Get lower and upper bounds of both metrics
    lower_max_infected = avg_max_infected - t_alpha * std_max_infected / np.sqrt(
        n_simulations
    )
    upper_max_infected = avg_max_infected + t_alpha * std_max_infected / np.sqrt(
        n_simulations
    )

    lower_max_infected_time = (
        avg_max_infected_time - t_alpha * std_max_infected_time / np.sqrt(n_simulations)
    )
    upper_max_infected_time = (
        avg_max_infected_time + t_alpha * std_max_infected_time / np.sqrt(n_simulations)
    )

    # Print nicely
    print("-"*85)
    print(f"{'CI    N[Vaccinated]='+str(n_vaccinated):<25}|{'Average':^20}|{'Lower':^20}|{'Upper':^20}")
    print("-"*85)
    print(
        f"{'Max Infected':<25}|{avg_max_infected:^20.2f}|{lower_max_infected:^20.2f}|{upper_max_infected:^20.2f}"
    )
    print(
        f"{'Time of Max Infection':<25}|{avg_max_infected_time:^20.2f}|{lower_max_infected_time:^20.2f}|{upper_max_infected_time:^20.2f}"
    )


def task1_g():
    # Set params
    n_simulations = 1000
    time_steps = 300
    population_size = 1000
    n_vaccinated_arr = [0, 100, 600, 800]
    n_infected = 50

    # Go through all the cases
    for n_vaccinated in n_vaccinated_arr:
        # Set y0
        y0 = np.array([population_size - n_vaccinated - n_infected, n_infected, n_vaccinated])
        calculate_CI(n_simulations, time_steps, y0, n_vaccinated)
        plot_realization(time_steps, y0, n_vaccinated)
        print("\n")


task1_g()
