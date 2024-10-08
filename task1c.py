import numpy as np
import scipy.stats as stats


def simulate(N: int = 7300) -> np.ndarray:
    alpha = 0.005
    beta = 0.01
    gamma = 0.1

    rng = np.random.default_rng()
    realizations = rng.uniform(0, 1, N)

    state = 0
    x = np.zeros(N, dtype=np.int64)

    for n in range(N):
        if state == 0:
            if realizations[n] <= beta:
                state = 1
        elif state == 1:
            if realizations[n] <= gamma:
                state = 2
        else:
            if realizations[n] <= alpha:
                state = 0
        x[n] = state

    return x


def confidence_interval_simulation(Nd: int, Ns: int):
    t = stats.t
    tail = int(Ns / 2)

    data = np.zeros((3, Nd))

    for i in range(Nd):
        x = simulate(Ns)
        _, counts = np.unique_counts(x[tail:])
        counts = counts / tail
        data[:, i] = counts
    avg = np.average(data, axis=1)
    std = np.std(data, axis=1, ddof=1)
    t_alpha = t.ppf(0.975, Nd-1)
    lower = avg - t_alpha * std / np.sqrt(Nd)
    upper = avg + t_alpha * std / np.sqrt(Nd)
    return (avg, std, lower, upper)


def task_c():
    Ns = 7300
    Nd = 30
    avg, std, lower, upper = confidence_interval_simulation(Nd, Ns)
    true_values = [10/31, 1/31, 20/31]
    for i in range(lower.size):
        print(f"State {i}: ({lower[i]:.4f}, {upper[i]:.4f})")
        print(f"True value: {true_values[i]:.4f}")
        print(f"Is contained: {lower[i] <= true_values[i] <= upper[i]}")
        print("-"*10)


if __name__ == "__main__":
    task_c()
