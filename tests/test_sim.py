# test_sim.py

import jax
import jax.numpy as jnp
import jsmfsb


def test_simts():
    lv = jsmfsb.models.lv()
    step = lv.step_gillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.sim_time_series(k0, lv.m, 0, 10, 0.1, step)
    assert out.shape == (100, 2)


def test_simsample():
    lv = jsmfsb.models.lv()
    step = lv.step_gillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.sim_sample(k0, 20, lv.m, 0, 10, step)
    assert out.shape == (20, 2)


def test_simsamples():
    lv = jsmfsb.models.lv()
    step = lv.step_gillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.sim_sample(k0, 20, lv.m, 0, 10, step)
    out_b = jsmfsb.sim_sample(k0, 20, lv.m, 0, 10, step, batch_size=5)
    assert jnp.all(out == out_b)


# eof
