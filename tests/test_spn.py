# test_spn.py

import jsmfsb
import jax



def test_spn():
    lv = jsmfsb.models.lv()
    step = lv.stepGillespie()
    k0 = jax.random.key(42)
    x1 = step(k0, lv.m, 0, 1)
    assert(x1.shape == (2,))





# eof

