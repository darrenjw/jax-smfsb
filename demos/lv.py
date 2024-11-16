#!/usr/bin/env python3
# lv.py
# use a pre-defined model

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import scipy as sp

import jsmfsb

lvmod = jsmfsb.models.lv()
step = lvmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, lvmod.m, 0, 30))

step_c = lvmod.step_cle(0.01)
print(step_c(k0, lvmod.m, 0, 30))

out = jsmfsb.sim_sample(k0, 10000, lvmod.m, 0, 30, step_c)
out = jnp.where(out > 1000, 1000, out)

print(sp.stats.describe(out))
fig, axes = plt.subplots(2, 1)
for i in range(2):
    axes[i].hist(out[:, i], bins=50)
fig.savefig("lv.pdf")

# eof
