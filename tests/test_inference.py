# test_inference.py
# tests relating to chapters 10 and 11

import jsmfsb
import jax
import jax.numpy as jnp
import jax.scipy as jsp



def test_metropolisHastings():
    key = jax.random.key(42)
    data = jax.random.normal(key, 250)*2 + 5
    llik = lambda x: jnp.sum(jsp.stats.norm.logpdf(data, x[0], x[1]))
    prop = lambda k,x: jax.random.normal(k, 2)*0.1 + x
    out = jsmfsb.metropolisHastings(key, jnp.array([1.0,1.0]), llik, prop,
                                   iters=1000, thin=2, verb=False)
    assert(out.shape == (1000, 2))



# eof

