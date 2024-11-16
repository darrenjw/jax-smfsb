#!/usr/bin/env python3
# m-mm.py
# use a pre-defined model

import jax
import matplotlib.pyplot as plt

import jsmfsb

mmmod = jsmfsb.models.mm()
step = mmmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, mmmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, mmmod.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
for i in range(4):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(mmmod.n)
fig.savefig("m-mm.pdf")

# eof
