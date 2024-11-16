#!/usr/bin/env python3
# abc_run.py

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

k0 = jax.random.key(42)
k1, k2 = jax.random.split(k0)

data = jax.random.normal(k1, 250) * 2 + 5


def rpr(k):
    return jnp.exp(jax.random.uniform(k, 2, minval=-3, maxval=3))


def rmod(k, th):
    return jax.random.normal(k, 250) * th[1] + th[0]


def sum_stats(dat):
    return jnp.array([jnp.mean(dat), jnp.std(dat)])


ssd = sum_stats(data)


def dist(ss):
    diff = ss - ssd
    return jnp.sqrt(jnp.sum(diff * diff))


def rdis(k, th):
    return dist(sum_stats(rmod(k, th)))


p, d = jsmfsb.abc_run(k2, 1000000, rpr, rdis)

q = jnp.quantile(d, 0.01)
prmat = jnp.vstack(p)
postmat = prmat[d < q, :]
its, var = postmat.shape

fig, axes = plt.subplots(3, 2)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].plot(postmat[:, 0], postmat[:, 1], linewidth=0.1)
axes[1, 0].plot(range(its), postmat[:, 0], linewidth=0.1)
axes[1, 1].plot(range(its), postmat[:, 1], linewidth=0.1)
axes[2, 0].hist(postmat[:, 0], bins=30)
axes[2, 1].hist(postmat[:, 1], bins=30)
fig.savefig("abc_run.pdf")


# eof
