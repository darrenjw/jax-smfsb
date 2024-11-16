#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models

N = 20
T = 30
x0 = jnp.zeros((2, N))
lv = jsmfsb.models.lv()
x0 = x0.at[:, int(N / 2)].set(lv.m)
step_lv_1d = lv.step_cle_1d(jnp.array([0.6, 0.6]))
k0 = jax.random.key(42)
x1 = step_lv_1d(k0, x0, 0, 1)
print(x1)
out = jsmfsb.sim_time_series_1d(k0, x0, 0, T, 1, step_lv_1d, True)
# print(out)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(out[i, :, :])
    axis.set_title(lv.n[i])
    fig.savefig(f"step_cle_1d{i}.pdf")


# eof
