import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

exp_dist = st.expon  # this is a general exponential distribution


def task_a():
    first_10 = []
    iters = 1000
    amount_claims = np.zeros(iters)
    for i in range(iters):
        t = 0
        times = [0, ]
        while t < 59:  # Exp dist gives time till next event, we want to find events until t >=59
            delta_t = exp_dist.rvs(0, 1 / 1.5)
            t += delta_t
            if t <= 59:
                times.append(t)

        if i < 10:
            first_10.append(times)

        amount_claims[i] = len(times) - 1

    amount_over = np.sum(amount_claims > 100) / iters

    print(f"Percentage of claims over 100: {amount_over}")
    for realization in first_10:
        plt.step(realization, np.arange(0, len(realization), 1))
    # arange is used since only the times of the claims is stored and we know the amount of claims increase by one for every event

    plt.xlabel("t[days]")
    plt.ylabel("Number of claims")
    plt.title("The 10 first realizations of X(t)")
    plt.savefig("plots/task2a.pdf")


task_a()
