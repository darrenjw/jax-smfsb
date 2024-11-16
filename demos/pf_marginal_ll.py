#!/usr/bin/env python3
# pf_marginal_ll.py

import jsmfsb
import jax
import jax.scipy as jsp
import jax.numpy as jnp


def obsll(x, t, y, th):
    return jnp.sum(jsp.stats.norm.logpdf(y - x, scale=10))


def sim_x(k, t0, th):
    k1, k2 = jax.random.split(k)
    return jnp.array([jax.random.poisson(k1, 50), jax.random.poisson(k2, 100)]).astype(
        jnp.float32
    )


def step(k, x, t, dt, th):
    sf = jsmfsb.models.lv(th).step_cle(0.1)
    # sf = jsmfsb.models.lv(th).step_gillespie()
    return sf(k, x, t, dt)


mll = jsmfsb.pf_marginal_ll(100, sim_x, 0, step, obsll, jsmfsb.data.lv_noise_10)

k = jax.random.split(jax.random.key(42), 5)
print(mll(k[0], jnp.array([1, 0.005, 0.6])))
print(mll(k[1], jnp.array([1, 0.005, 0.6])))
print(mll(k[2], jnp.array([1, 0.005, 0.6])))


print(mll(k[3], jnp.array([1, 0.005, 0.5])))
print(mll(k[4], jnp.array([1, 0.005, 0.5])))
