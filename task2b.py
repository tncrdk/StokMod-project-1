import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt

exp_dist = st.expon


def task2b():
    first_10_times = []
    first_10_amounts = []

    amount_claims = np.zeros(1000)
    claim_amounts = np.zeros(1000)

    for i in range(1000):
        t = 0
        times = [0]
        amounts = [0]
        while t < 59:
            delta_t = exp_dist.rvs(0, 1 / 1.5)  # realisation of an exponential variable
            # print(delta_t)
            t += delta_t
            if t <= 59:
                times.append(t)
                amounts.append(exp_dist.rvs(0, 1 / 10) + amounts[-1])

        amount_claims[i] = len(times)

        claim_amounts[i] = amounts[-1]

        if i < 10:
            first_10_times.append(times)
            first_10_amounts.append(amounts)

    prob_over_8 = np.sum(claim_amounts > 8) / 1000
    print(prob_over_8)

    for k in range(10):
        plt.step(first_10_times[k], first_10_amounts[k])

    plt.title("10 first realisations of Z(t)")
    plt.xlabel("t")
    plt.ylabel("Claim amount in millions")
    plt.show()


task2b()
