#!/usr/bin/env python3
# lv.py
# use a pre-defined model

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl
from jax import grad, jit

from spn import Spn

from models import *
from sim import *




lvmod = lv()
step = lvmod.stepGillespie()
k0 = jax.random.key(42)
print(step(k0, lvmod.m, 0, 30))

stepC = lvmod.stepCLE(0.01)
print(stepC(k0, lvmod.m, 0, 30))


# simTs
out = simTs(k0, lvmod.m, 0, 30, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(lvmod.n)
fig.savefig("lv.pdf")

# simSample
out = simSample(k0, 10000, lvmod.m, 0, 30, stepC)
out = jnp.where(out > 1000, 1000, out)
import scipy as sp
print(sp.stats.describe(out))
fig, axes = plt.subplots(2,1)
for i in range(2):
    axes[i].hist(out[:,i], bins=50)
fig.savefig("lvH.pdf")

# eof

