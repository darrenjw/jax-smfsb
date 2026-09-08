# mcmc.py

# Some additional functions for MCMC output processing.

# Not in the main "jsmfsb" package only because they have
# additional dependencies.

# Note that this is not JAX-based - it's just post-processing of MCMC output.

# See "pmmh.py" demo for illustration of use


import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def acf(x, lag_max):
    return np.array([1] + [np.corrcoef(x[:-i], x[i:])[0, 1] for i in range(1, lag_max)])


def mcmc_summary(
    mat,
    file_name="mcmc.pdf",
    labels=False,
    truth=False,
    bins=30,
    lag_max=100,
    show=True,
    plot=True,
):
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
        fig.tight_layout()
        for i in range(p):
            axes[i, 0].plot(range(n), mat[:, i], linewidth=0.4)
            axes[i, 0].set_title("Traceplot")
            if labels:
                axes[i, 0].set_ylabel(labels[i])
            if truth:
                axes[i, 0].hlines(truth[i], 0, n, "r")
            axes[i, 1].plot(range(lag_max), acf(mat[:, i], lag_max))
            axes[i, 1].set_ylim([-0.5, 1])
            axes[i, 1].set_title("ACF")
            axes[i, 1].axhline(y=0, color="g", linewidth=0.6)
            h, b, bc = axes[i, 2].hist(mat[:, i], bins=bins, density=True)
            axes[i, 2].set_title("Density")
            if truth:
                axes[i, 2].vlines(truth[i], 0, np.max(h), "r")
        fig.savefig(file_name, dpi=300, bbox_inches="tight")
    return summ


def pairs(mat, file_name="pairs.pdf", labels=False, truth=False, bins=30):
    n, p = mat.shape
    summ = sp.stats.describe(mat)
    fig, axes = plt.subplots(p, p)
    fig.tight_layout()
    for row in range(p):
        for col in range(p):
            if row == col:
                h, b, bc = axes[row, col].hist(mat[:, row], bins, density=True)
                if truth:
                    axes[row, col].vlines(truth[row], 0, np.max(h), "r")
            else:
                axes[row, col].scatter(mat[:, col], mat[:, row], marker=".")
                if truth:
                    axes[row, col].vlines(
                        truth[col], summ.minmax[0][row], summ.minmax[1][row], "r"
                    )
                    axes[row, col].hlines(
                        truth[row], summ.minmax[0][col], summ.minmax[1][col], "r"
                    )
            if (col == 0) and labels:
                axes[row, col].set_ylabel(labels[row])
            if (row == p - 1) and labels:
                axes[row, col].set_xlabel(labels[col])
    fig.savefig(file_name, dpi=300, bbox_inches="tight")
    return fig


# eof
