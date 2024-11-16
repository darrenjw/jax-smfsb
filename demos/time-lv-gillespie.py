#!/usr/bin/env python3
# time-lv-gillespie.py
# time the gillespie algorithm

import jax
import jax.numpy as jnp
import jsmfsb
import matplotlib.pyplot as plt
import scipy as sp
import time

lvmod = jsmfsb.models.lv()
step = lvmod.step_gillespie()
k0 = jax.random.key(42)

## Start timer
start_time = time.time()
out = jsmfsb.sim_sample(k0, 10000, lvmod.m, 0, 20, step, batch_size=100)
end_time = time.time()
## End timer
elapsed_time = end_time - start_time
print(f"\n\nElapsed time: {elapsed_time} seconds\n\n")

out = jnp.where(out > 1000, 1000, out)

print(sp.stats.describe(out))
fig, axes = plt.subplots(2, 1)
for i in range(2):
    axes[i].hist(out[:, i], bins=50)
fig.savefig("time-lv-gillespie.pdf")


# eof
