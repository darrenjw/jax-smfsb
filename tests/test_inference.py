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

    
def test_pfmllik():
    def obsll(x, t, y, th):
        return jnp.sum(jsp.stats.norm.logpdf((y-x)/10))
    def simX(k, t0, th):
        k1, k2 = jax.random.split(k)
        return jnp.array([jax.random.poisson(k1, 50),
                          jax.random.poisson(k2, 100)]).astype(jnp.float32)
    def step(k, x, t, dt, th):
        sf = jsmfsb.models.lv(th).stepCLE()
        return sf(k, x, t, dt)
    mll = jsmfsb.pfMLLik(50, simX, 0, step, obsll, jsmfsb.data.LVnoise10)
    k = jax.random.key(42)
    assert (mll(k, jnp.array([1, 0.005, 0.6])) > mll(k, jnp.array([2, 0.005, 0.6])))


# eof

