#!/usr/bin/env python3
# m-id.py
# use a pre-defined model

import jax
import matplotlib.pyplot as plt

import jsmfsb

idmod = jsmfsb.models.id()
step = idmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, idmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, idmod.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
for i in range(1):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(idmod.n)
fig.savefig("m-id.pdf")

# eof
