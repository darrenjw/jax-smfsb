# mcmc.py

# Some additional functions for MCMC output processing.

# Not in the main "smfsb" package only because they have
# additional dependencies.

# Note that this is not JAX-based - it's just post-processing of MCMC output.

# See "pmmh.py" demo for illustration of use


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def acf(x, lag_max):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lag_max)])


def mcmc_summary(mat, file_name="mcmc.pdf", bins=30, lag_max=100, show=True, plot=True):
    n, p = mat.shape
    summ = sp.stats.describe(mat)
    med = np.median(mat, 0)
    if show:
        print(f"Mean: {summ.mean}")
        print(f"Median: {med}")
        print(f"Variance: {summ.variance}")
        print(f"SDs: {np.sqrt(summ.variance)}")
        print(f"Min: {summ.minmax[0]}")
        print(f"Max: {summ.minmax[1]}")
    if plot:
        fig, axes = plt.subplots(p, 3)
        for i in range(p):
            axes[i, 0].plot(range(n), mat[:, i], linewidth=0.1)
            axes[i, 1].plot(range(lag_max), acf(mat[:, i], lag_max))
            axes[i, 1].set_ylim([-0.5, 1])
            axes[i, 1].axhline(y=0, color="g", linewidth=0.5)
            axes[i, 2].hist(mat[:, i], bins=bins)
        fig.savefig(file_name)
    return summ


# eof
