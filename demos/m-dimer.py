#!/usr/bin/env python3
# m-dimer.py
# use a pre-defined model

import jax
import matplotlib.pyplot as plt

import jsmfsb

dimermod = jsmfsb.models.dimer()
step = dimermod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, dimermod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, dimermod.m, 0, 30, 0.1, step)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(dimermod.n)
fig.savefig("m-dimer.pdf")

# eof
