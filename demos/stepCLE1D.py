#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models

N=20
T=30
x0 = jnp.zeros((2,N))
lv = jsmfsb.models.lv()
x0 = x0.at[:,int(N/2)].set(lv.m)
stepLv1d = lv.step_cle1D(jnp.array([0.6, 0.6]))
k0 = jax.random.key(42)
x1 = stepLv1d(k0, x0, 0, 1)
print(x1)
out = jsmfsb.simTs1D(k0, x0, 0, T, 1, stepLv1d, True)
#print(out)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(out[i,:,:])
    axis.set_title(lv.n[i])
    fig.savefig(f"step_cle1D{i}.pdf")


# eof
