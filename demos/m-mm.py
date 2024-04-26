#!/usr/bin/env python3
# m-mm.py
# use a pre-defined model

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl
from jax import grad, jit

import jsmfsb

mmmod = jsmfsb.models.mm()
step = mmmod.stepGillespie()
k0 = jax.random.key(42)
print(step(k0, mmmod.m, 0, 30))

out = jsmfsb.simTs(k0, mmmod.m, 0, 100, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(4):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(mmmod.n)
fig.savefig("m-mm.pdf")

# eof

