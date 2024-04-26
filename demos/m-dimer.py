#!/usr/bin/env python3
# m-dimer.py
# use a pre-defined model

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl
from jax import grad, jit

import jsmfsb

dimermod = jsmfsb.models.dimer()
step = dimermod.stepGillespie()
k0 = jax.random.key(42)
print(step(k0, dimermod.m, 0, 30))

out = jsmfsb.simTs(k0, dimermod.m, 0, 30, 0.1, step)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(dimermod.n)
fig.savefig("m-dimer.pdf")

# eof

