#!/usr/bin/env python3

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models

M=200
N=250
T=30
x0 = jnp.zeros((2,M,N))
lv = jsmfsb.models.lv()
x0 = x0.at[:,int(M/2),int(N/2)].set(lv.m)
stepLv2d = lv.stepCLE2D(jnp.array([0.6, 0.6]), 0.1)
k0 = jax.random.key(42)
x1 = stepLv2d(k0, x0, 0, T)

fig, axis = plt.subplots()
for i in range(2):
    axis.imshow(x1[i,:,:])
    axis.set_title(lv.n[i])
    fig.savefig(f"stepCLE2Df{i}.pdf")


    
    
# eof