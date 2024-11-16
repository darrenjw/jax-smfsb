#!/usr/bin/env python3
# metropolis_hastings.py

import jsmfsb
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

k0 = jax.random.key(42)
k1, k2 = jax.random.split(k0)
data = jax.random.normal(k1, 250) * 2 + 5


def llik(k, x):
    return jnp.sum(jsp.stats.norm.logpdf(data, x[0], x[1]))


def prop(k, x):
    return jax.random.normal(k, 2) * 0.1 + x


postmat = jsmfsb.metropolis_hastings(k2, jnp.array([1.0, 1.0]), llik, prop, verb=False)

fig, axes = plt.subplots(3, 2)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].plot(postmat[:, 0], postmat[:, 1], linewidth=0.1)
axes[1, 0].plot(range(10000), postmat[:, 0], linewidth=0.1)
axes[1, 1].plot(range(10000), postmat[:, 1], linewidth=0.1)
axes[2, 0].hist(postmat[:, 0], bins=30)
axes[2, 1].hist(postmat[:, 1], bins=30)
fig.savefig("metropolis_hastings.pdf")


# eof
