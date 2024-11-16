# test_spatial.py
# tests relating to chapter 9

import jsmfsb
import jax
import jax.numpy as jnp
import jsmfsb.models


def test_step_gillespie_1d():
    n = 20
    x0 = jnp.zeros((2, n))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(n / 2)].set(lv.m)
    step_lv_1d = lv.step_gillespie_1d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = step_lv_1d(k0, x0, 0, 1)
    assert x1.shape == (2, n)


def test_sim_time_series_1d():
    n = 8
    tt = 6
    x0 = jnp.zeros((2, n))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(n / 2)].set(lv.m)
    step_lv_1d = lv.step_gillespie_1d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    out = jsmfsb.sim_time_series_1d(k0, x0, 0, tt, 1, step_lv_1d)
    assert out.shape == (2, n, tt + 1)


def test_step_gillespie_2d():
    m = 16
    n = 20
    x0 = jnp.zeros((2, m, n))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(m / 2), int(n / 2)].set(lv.m)
    step_lv_2d = lv.step_gillespie_2d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = step_lv_2d(k0, x0, 0, 1)
    assert x1.shape == (2, m, n)


def test_sim_time_series_2d():
    m = 16
    n = 20
    x0 = jnp.zeros((2, m, n))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(m / 2), int(n / 2)].set(lv.m)
    step_lv_2d = lv.step_gillespie_2d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    out = jsmfsb.sim_time_series_2d(k0, x0, 0, 5, 1, step_lv_2d)
    assert out.shape == (2, m, n, 6)


def test_step_cle_1d():
    n = 20
    x0 = jnp.zeros((2, n))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(n / 2)].set(lv.m)
    step_lv_1d = lv.step_cle_1d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = step_lv_1d(k0, x0, 0, 1)
    assert x1.shape == (2, n)


def test_step_cle_2d():
    m = 16
    n = 20
    x0 = jnp.zeros((2, m, n))
    lv = jsmfsb.models.lv()
    x0 = x0.at[:, int(m / 2), int(n / 2)].set(lv.m)
    step_lv_2d = lv.step_cle_2d(jnp.array([0.6, 0.6]))
    k0 = jax.random.key(42)
    x1 = step_lv_2d(k0, x0, 0, 1)
    assert x1.shape == (2, m, n)


# eof
