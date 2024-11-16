#!/usr/bin/env python3
# m-lv-euler.py
# use a pre-defined model

import jax
import matplotlib.pyplot as plt

import jsmfsb

lvmod = jsmfsb.models.lv()
step = lvmod.step_euler(0.001)
k0 = jax.random.key(42)
print(step(k0, lvmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, lvmod.m, 0, 30, 0.1, step)

fig, axis = plt.subplots()
for i in range(2):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(lvmod.n)
fig.savefig("m-lv-euler.pdf")

# eof
