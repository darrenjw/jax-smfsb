# test_spatial.py
# tests relating to chapter 9

import jsmfsb
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jsmfsb.models

def test_step_gillespie1D():
    N=20
    x0 = jnp.zeros((2,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:,int(N/2)].set(lv.m)
    stepLv1d = lv.step_gillespie1D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = stepLv1d(k0, x0, 0, 1)
    assert(x1.shape == (2,N))

def test_simTs1D():
    N=8
    T=6
    x0 = jnp.zeros((2,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:,int(N/2)].set(lv.m)
    stepLv1d = lv.step_gillespie1D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    out = jsmfsb.simTs1D(k0, x0, 0, T, 1, stepLv1d)
    assert(out.shape == (2, N, T+1))

def test_step_gillespie2D():
    M=16
    N=20
    x0 = jnp.zeros((2,M,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(M/2), int(N/2)].set(lv.m)
    stepLv2d = lv.step_gillespie2D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = stepLv2d(k0, x0, 0, 1)
    assert(x1.shape == (2, M, N))

def test_simTs2D():
    M=16
    N=20
    x0 = jnp.zeros((2,M,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:,int(M/2),int(N/2)].set(lv.m)
    stepLv2d = lv.step_gillespie2D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    out = jsmfsb.simTs2D(k0, x0, 0, 5, 1, stepLv2d)
    assert(out.shape == (2, M, N, 6))

def test_step_cle1D():
    N=20
    x0 = jnp.zeros((2,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:,int(N/2)].set(lv.m)
    stepLv1d = lv.step_cle1D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = stepLv1d(k0, x0, 0, 1)
    assert(x1.shape == (2, N))

def test_step_cle2D():
    M=16
    N=20
    x0 = jnp.zeros((2,M,N))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:,int(M/2),int(N/2)].set(lv.m)
    stepLv2d = lv.step_cle2D(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = stepLv2d(k0, x0, 0, 1)
    assert(x1.shape == (2, M, N))





# eof

