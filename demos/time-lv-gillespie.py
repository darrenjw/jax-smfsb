#!/usr/bin/env python3
# time-lv-gillespie.py
# time the gillespie algorithm

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl
from jax import grad, jit
import jsmfsb
import matplotlib.pyplot as plt
import time

lvmod = jsmfsb.models.lv()
step = lvmod.stepGillespie()
k0 = jax.random.key(42)

## Start timer
startTime = time.time()
#out = jsmfsb.simSample(k0, 10000, lvmod.m, 0, 20, step)
out = jsmfsb.simSampleMap(k0, 10000, lvmod.m, 0, 20, step)
endTime = time.time()
## End timer
elapsedTime = endTime - startTime
print(f"\n\nElapsed time: {elapsedTime} seconds\n\n")

out = jnp.where(out > 1000, 1000, out)
import scipy as sp
print(sp.stats.describe(out))
fig, axes = plt.subplots(2,1)
for i in range(2):
    axes[i].hist(out[:,i], bins=50)
fig.savefig("time-lv-gillespie.pdf")


# eof

