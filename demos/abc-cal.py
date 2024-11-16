#!/usr/bin/env python3
# abc-cal.py
# ABC with calibrated summary stats

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

print("ABC with calibrated summary stats")

data = jsmfsb.data.lv_perfect[:, 1:3]


def rpr(k):
    k1, k2, k3 = jax.random.split(k, 3)
    return jnp.exp(
        jnp.array(
            [
                jax.random.uniform(k1, minval=-3, maxval=3),
                jax.random.uniform(k2, minval=-8, maxval=-2),
                jax.random.uniform(k3, minval=-4, maxval=2),
            ]
        )
    )


def rmod(k, th):
    return jsmfsb.sim_time_series(
        k, jnp.array([50.0, 100.0]), 0, 30, 2, jsmfsb.models.lv(th).step_cle(0.1)
    )


def ss1d(vec):
    n = len(vec)
    mean = jnp.nanmean(vec)
    v0 = vec - mean
    var = jnp.nanvar(v0)
    acs = [
        jnp.corrcoef(v0[0 : (n - 1)], v0[1:n])[0, 1],
        jnp.corrcoef(v0[0 : (n - 2)], v0[2:n])[0, 1],
        jnp.corrcoef(v0[0 : (n - 3)], v0[3:n])[0, 1],
    ]
    # print(mean)
    # print(var)
    # print(acs)
    return jnp.array([jnp.log(mean + 1), jnp.log(var + 1), acs[0], acs[1], acs[2]])


def ssi(ts):
    return jnp.concatenate(
        (
            ss1d(ts[:, 0]),
            ss1d(ts[:, 1]),
            jnp.array([jnp.corrcoef(ts[:, 0], ts[:, 1])[0, 1]]),
        )
    )


print("Pilot run")
k0 = jax.random.key(42)
k1, k2 = jax.random.split(k0)

p, d = jsmfsb.abc_run(k1, 100000, rpr, lambda k, th: ssi(rmod(k, th)), batch_size=10000)
prmat = jnp.vstack(p)
dmat = jnp.vstack(d)
print(prmat.shape)
print(dmat.shape)
dmat = dmat.at[dmat == jnp.inf].set(jnp.nan)
sds = jnp.nanstd(dmat, 0)
print(sds)


print("Main run with calibrated summary stats")


def sum_stats(dat):
    return ssi(dat) / sds


ssd = sum_stats(data)


def dist(ss):
    diff = ss - ssd
    return jnp.sqrt(jnp.sum(diff * diff))


def rdis(k, th):
    return dist(sum_stats(rmod(k, th)))


p, d = jsmfsb.abc_run(k2, 1000000, rpr, rdis, batch_size=100000, verb=False)

q = jnp.nanquantile(d, 0.01)
prmat = jnp.vstack(p)
postmat = prmat[d < q, :]
its, var = postmat.shape
print(its, var)

postmat = jnp.log(postmat)  # look at posterior on log scale

fig, axes = plt.subplots(2, 3)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].scatter(postmat[:, 0], postmat[:, 2], s=0.5)
axes[0, 2].scatter(postmat[:, 1], postmat[:, 2], s=0.5)
axes[1, 0].hist(postmat[:, 0], bins=30)
axes[1, 1].hist(postmat[:, 1], bins=30)
axes[1, 2].hist(postmat[:, 2], bins=30)
fig.savefig("abc-cal.pdf")

print("All done.")


# eof
