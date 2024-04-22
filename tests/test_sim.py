# test_sim.py

import jax
import jax.numpy as jnp
import jsmfsb

def test_simts():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.simTs(k0, lv.m, 0, 10, 0.1, step)
    assert(out.shape == (100, 2))

def test_simsample():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.simSample(k0, 20, lv.m, 0, 10, step)
    assert(out.shape == (20, 2))

def test_simsamplemap():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.simSampleMap(k0, 20, lv.m, 0, 10, step)
    assert(out.shape == (20, 2))

def test_simsamples():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k0 = jax.random.key(42)
    out = jsmfsb.simSample(k0, 20, lv.m, 0, 10, step)
    outM = jsmfsb.simSampleMap(k0, 20, lv.m, 0, 10, step)
    assert(jnp.all(out == outM))




# eof

