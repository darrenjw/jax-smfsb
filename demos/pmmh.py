#!/usr/bin/env python3
# pmmh.py
# PMMH demo
# Particle marginal Metropolis-Hastings

import jsmfsb
import mcmc  # extra functions in the demo directory
import jax
import jax.scipy as jsp
import jax.numpy as jnp

print("PMMH")


def obsll(x, t, y, th):
    return jnp.sum(jsp.stats.norm.logpdf(y - x, scale=10))


def sim_x(k, t0, th):
    k1, k2 = jax.random.split(k)
    return jnp.array([jax.random.poisson(k1, 50), jax.random.poisson(k2, 100)]).astype(
        jnp.float32
    )


def step(k, x, t, dt, th):
    # sf = jsmfsb.models.lv(th).step_gillespie()
    sf = jsmfsb.models.lv(th).step_cle(0.1)
    return sf(k, x, t, dt)


mll = jsmfsb.pf_marginal_ll(100, sim_x, 0, step, obsll, jsmfsb.data.lv_noise_10)

print("Test evals")
k0 = jax.random.key(42)
k1, k2, k3 = jax.random.split(k0, 3)
print(mll(k1, jnp.array([1, 0.005, 0.6])))
print(mll(k2, jnp.array([1, 0.005, 0.5])))

print("Now the main MCMC loop")


def prop(k, th, tune=0.01):
    return jnp.exp(jax.random.normal(k, shape=(3)) * tune) * th


thmat = jsmfsb.metropolis_hastings(
    k3, jnp.array([1, 0.005, 0.6]), mll, prop, iters=5000, thin=1, verb=False
)

print("MCMC done. Now processing the results...")

mcmc.mcmc_summary(thmat, "pmmh.pdf")

print("All finished.")

# eof
