# test_models.py

import jsmfsb
import jax


def test_bd():
    bd = jsmfsb.models.bd()
    step = bd.stepGillespie()
    k = jax.random.key(42)
    x = step(k, bd.m, 0, 1)
    assert(x[0] <= bd.m[0])

def test_lv():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k = jax.random.key(42)
    assert(step(k, lv.m, 0, 1).shape == lv.m.shape)



# eof

