#!/usr/bin/env python3
# m-sir.py
# use a pre-defined model

import jax
import matplotlib.pyplot as plt

import jsmfsb

sirmod = jsmfsb.models.sir()
step = sirmod.step_gillespie()
k0 = jax.random.key(42)
print(step(k0, sirmod.m, 0, 30))

out = jsmfsb.sim_time_series(k0, sirmod.m, 0, 100, 0.1, step)

fig, axis = plt.subplots()
for i in range(3):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(sirmod.n)
fig.savefig("m-sir.pdf")

# eof
