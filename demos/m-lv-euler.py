#!/usr/bin/env python3
# m-lv-euler.py
# use a pre-defined model

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl
from jax import grad, jit

import jsmfsb

lvmod = jsmfsb.models.lv()
step = lvmod.stepEuler(0.001)
k0 = jax.random.key(42)
print(step(k0, lvmod.m, 0, 30))

out = jsmfsb.simTs(k0, lvmod.m, 0, 30, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(lvmod.n)
fig.savefig("m-lv-euler.pdf")

# eof

