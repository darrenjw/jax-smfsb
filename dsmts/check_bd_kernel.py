# check_bd_kernel.py

# Custom script to check that the kernel for gillespie simulation
#   of the birth-death process has the correct mean and variance

# Based on 00001/dsmts-001-01 (first time point)
# Initial marking 100. Birth rate 0.1. Death rate 0.11.
# After 1 time unit, the mean should be 99.00498 and the
#   sd should be 4.54834.


# NOTE: this script was written to identify a bug, which turned out to be
#   in sim_time_series, and has since been fixed. This code is now for
#   historical interest only. Will be removed from the repo at some point.

import time
import jax
import jsmfsb
import jax.numpy as jnp

N = 1000
true_mean = 99.00498
true_sd = 4.54834

print("Running with N=" + str(N))
bd = jsmfsb.models.bd([0.1, 0.11])
step = bd.step_gillespie()
k0 = jax.random.key(42)
out = jsmfsb.sim_sample(k0, N, bd.m, 0, 1, step)
sample_mean = jnp.mean(out)
print(sample_mean)
# Is this close enough?
z_score = jnp.sqrt(N) * (sample_mean - true_mean) / true_sd
print(z_score)
# Kernel, according to sim_sample, seems absolutely fine.
# Do sd test, anyway, just to make sure.
sxx = out - true_mean
sxx = sxx * sxx
sxx = jnp.sum(sxx)
sts = sxx / N
print(jnp.sqrt(sts))
y_score = (sts / (true_sd * true_sd) - 1) * jnp.sqrt(N / 2)
print(y_score)
# Again, seems fine.
time.sleep(5)

# Is the issue with sim_time_series somehow?
sx = 0
sxx = 0
keys = jax.random.split(k0, N)
for k in keys:
    out = jsmfsb.sim_time_series(k, bd.m, 0, 2, 1, step)
    assert out.shape == (3, 1)
    sx = sx + out[1, 0]
    si = out[1, 0] - true_mean
    sxx = sxx + (si * si)
sample_mean = sx / N
print(sample_mean)
z_score = jnp.sqrt(N) * (sample_mean - true_mean) / true_sd
print(z_score)
sts = sxx / N
print(jnp.sqrt(sts))
y_score = (sts / (true_sd * true_sd) - 1) * jnp.sqrt(N / 2)
print(y_score)
# Yes! Somehow a problem using sim_time_series...
# Why?! Because the initial state isn't included in the output,
# so the output is shifted by one relative to what it should be!

# NOTE: this bug is now fixed, and the tests are all fine.


# eof
