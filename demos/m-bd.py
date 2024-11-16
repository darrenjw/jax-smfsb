#!/usr/bin/env python3
# m-bd.py
# use a pre-defined model

import jax
import matplotlib.pyplot as plt

import jsmfsb

bdmod = jsmfsb.models.bd()
step = bdmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, bdmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, bdmod.m, 0, 20, 0.1, step)

fig, axis = plt.subplots()
for i in range(1):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(bdmod.n)
fig.savefig("m-bd.pdf")

# eof
