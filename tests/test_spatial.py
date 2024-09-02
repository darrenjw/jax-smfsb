# test_spatial.py
# tests relating to chapter 9

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models

def test_stepGillespie1D():
    N=20
    x0 = jnp.zeros((2,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:,int(N/2)].set(lv.m)
    stepLv1d = lv.stepGillespie1D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = stepLv1d(k0, x0, 0, 1)
    assert(x1.shape == (2,N))










# eof

