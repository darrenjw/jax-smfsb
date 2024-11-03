# test_models.py

import jsmfsb
import jax
import jax.numpy as jnp


def test_bd():
    bd = jsmfsb.models.bd()
    step = bd.step_gillespie()
    k = jax.random.key(42)
    x = step(k, bd.m, 0, 10)
    assert x[0] <= bd.m[0]
    assert x[0] >= 0


def test_dimer():
    dimer = jsmfsb.models.dimer()
    step = dimer.step_gillespie()
    k = jax.random.key(42)
    x = step(k, dimer.m, 0, 10)
    assert x.shape == dimer.m.shape
    assert jnp.min(x) >= 0
    assert x[0] + 2 * x[1] == dimer.m[0] + 2 * dimer.m[1]


def test_id():
    id = jsmfsb.models.id()
    step = id.step_gillespie()
    k = jax.random.key(42)
    x = step(k, id.m, 0, 10)
    assert x[0] >= 0


def test_lv():
    lv = jsmfsb.models.lv()
    step = lv.step_gillespie()
    k = jax.random.key(42)
    x = step(k, lv.m, 0, 10)
    assert x.shape == lv.m.shape
    assert jnp.min(x) >= 0


def test_mm():
    mm = jsmfsb.models.mm()
    step = mm.step_gillespie()
    k = jax.random.key(42)
    x = step(k, mm.m, 0, 10)
    assert x.shape == mm.m.shape
    assert jnp.min(x) >= 0
    assert x[1] + x[2] == mm.m[1] + mm.m[2]
    assert x[0] + x[2] + x[3] == mm.m[0] + mm.m[2] + mm.m[3]


def test_sir():
    sir = jsmfsb.models.sir()
    step = sir.step_gillespie()
    k = jax.random.key(42)
    x = step(k, sir.m, 0, 10)
    assert x.shape == sir.m.shape
    assert jnp.min(x) >= 0
    assert jnp.sum(x) == jnp.sum(sir.m)


# eof
