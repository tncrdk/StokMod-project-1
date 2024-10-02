import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

exp_dist = st.expon


def task_a():
    first_10 = []

    amount_claims = np.zeros(1000)
    for i in range(1000):
        t = 0
        times = [0, ]
        while t < 59:
            delta_t = exp_dist.rvs(0, 1 / 1.5)
            # print(delta_t)
            t += delta_t
            if t <= 59:
                times.append(t)

        if i < 10:
            first_10.append(times)

        amount_claims[i] = len(times)

    print(amount_claims)
    amount_over = np.sum(amount_claims > 100) / 1000

    print(amount_over)
    for realization in first_10:
        plt.step(realization, np.arange(0, len(realization), 1))

    plt.xlabel("t")
    plt.ylabel("X(t)")
    plt.title("10 first realizations of X(t)")
    plt.show()


task_a()
