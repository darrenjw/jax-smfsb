# test_spn.py

import jsmfsb
import jax
import jax.numpy as jnp


def test_spn():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k0 = jax.random.key(42)
    x1 = step(k0, lv.m, 0, 1)
    assert(x1.shape == (2,))
    assert(jnp.min(x1) >= 0)

def test_cle():
    lv = jsmfsb.models.lv()
    step = lv.stepCLE(0.001)
    k0 = jax.random.key(42)
    x1 = step(k0, lv.m, 0, 1)
    assert(x1.shape == (2,))
    assert(jnp.min(x1) > 0.0)

    



# eof

