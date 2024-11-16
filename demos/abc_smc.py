#!/usr/bin/env python3
# abc_smc.py

import jsmfsb
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import matplotlib.pyplot as plt

print("ABC-SMC")

data = jsmfsb.data.lv_perfect[:, 1:3]

# Very vague prior


def rpr0(k):
    k1, k2, k3 = jax.random.split(k, 3)
    return jnp.array(
        [
            jax.random.uniform(k1, minval=-3, maxval=3),
            jax.random.uniform(k2, minval=-8, maxval=-2),
            jax.random.uniform(k3, minval=-4, maxval=2),
        ]
    )


def dpr0(th):
    return jnp.sum(
        jnp.log(
            jnp.array(
                [
                    ((th[0] > -3) & (th[0] < 3)) / 6,
                    ((th[1] > -8) & (th[1] < -2)) / 6,
                    ((th[2] > -4) & (th[2] < 2)) / 6,
                ]
            )
        )
    )


# Slightly less vague prior


def rpr(k):
    k1, k2, k3 = jax.random.split(k, 3)
    return jnp.array(
        [
            jax.random.uniform(k1, minval=-2, maxval=2),
            jax.random.uniform(k2, minval=-7, maxval=-3),
            jax.random.uniform(k3, minval=-3, maxval=1),
        ]
    )


def dpr(th):
    return jnp.sum(
        jnp.log(
            jnp.array(
                [
                    ((th[0] > -2) & (th[0] < 2)) / 4,
                    ((th[1] > -7) & (th[1] < -3)) / 4,
                    ((th[2] > -3) & (th[2] < 1)) / 4,
                ]
            )
        )
    )


# Model


def rmod(k, th):
    return jsmfsb.sim_time_series(
        k, jnp.array([50.0, 100]), 0, 30, 2, jsmfsb.models.lv(jnp.exp(th)).step_cle(0.1)
    )


print("Pilot run...")


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
    return jnp.array([jnp.log(mean + 1), jnp.log(var + 1), acs[0], acs[1], acs[2]])


def ssi(ts):
    return jnp.concatenate(
        (
            ss1d(ts[:, 0]),
            ss1d(ts[:, 1]),
            jnp.array([jnp.corrcoef(ts[:, 0], ts[:, 1])[0, 1]]),
        )
    )


key = jax.random.key(42)
p, d = jsmfsb.abc_run(key, 20000, rpr, lambda k, th: ssi(rmod(k, th)), verb=False)
prmat = jnp.vstack(p)
dmat = jnp.vstack(d)
print(prmat.shape)
print(dmat.shape)
dmat = dmat.at[dmat == jnp.inf].set(jnp.nan)
sds = jnp.nanstd(dmat, 0)
print(sds)


def sum_stats(dat):
    return ssi(dat) / sds


ssd = sum_stats(data)

print("Main ABC-SMC run")


def dist(ss):
    diff = ss - ssd
    return jnp.sqrt(jnp.sum(diff * diff))


def rdis(k, th):
    return dist(sum_stats(rmod(k, th)))


def rper(k, th):
    return th + jax.random.normal(k, 3) * 0.5


def dper(ne, ol):
    return jnp.sum(jsp.stats.norm.logpdf(ne, ol, 0.5))


postmat = jsmfsb.abc_smc(
    key, 10000, rpr, dpr, rdis, rper, dper, factor=5, steps=8, verb=True
)

its, var = postmat.shape
print(its, var)

fig, axes = plt.subplots(2, 3)
axes[0, 0].scatter(postmat[:, 0], postmat[:, 1], s=0.5)
axes[0, 1].scatter(postmat[:, 0], postmat[:, 2], s=0.5)
axes[0, 2].scatter(postmat[:, 1], postmat[:, 2], s=0.5)
axes[1, 0].hist(postmat[:, 0], bins=30)
axes[1, 1].hist(postmat[:, 1], bins=30)
axes[1, 2].hist(postmat[:, 2], bins=30)
fig.savefig("abc_smc.pdf")

print("All done.")


# eof
