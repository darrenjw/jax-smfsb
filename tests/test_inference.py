# test_inference.py
# tests relating to chapters 10 and 11

import jsmfsb
import jax
import jax.numpy as jnp
import jax.scipy as jsp


def test_metropolis_hastings():
    key = jax.random.key(42)
    data = jax.random.normal(key, 250) * 2 + 5
    llik = lambda k, x: jnp.sum(jsp.stats.norm.logpdf(data, x[0], x[1]))
    prop = lambda k, x: jax.random.normal(k, 2) * 0.1 + x
    out = jsmfsb.metropolis_hastings(
        key, jnp.array([1.0, 1.0]), llik, prop, iters=1000, thin=2, verb=False
    )
    assert out.shape == (1000, 2)


def test_pfmllik():
    def obsll(x, t, y, th):
        return jnp.sum(jsp.stats.norm.logpdf((y - x) / 10))

    def sim_x(k, t0, th):
        k1, k2 = jax.random.split(k)
        return jnp.array(
            [jax.random.poisson(k1, 50), jax.random.poisson(k2, 100)]
        ).astype(jnp.float32)

    def step(k, x, t, dt, th):
        sf = jsmfsb.models.lv(th).step_cle()
        return sf(k, x, t, dt)

    mll = jsmfsb.pf_marginal_ll(50, sim_x, 0, step, obsll, jsmfsb.data.lv_noise_10)
    k = jax.random.key(42)
    assert mll(k, jnp.array([1, 0.005, 0.6])) > mll(k, jnp.array([2, 0.005, 0.6]))


def test_abc_run():
    k0 = jax.random.key(42)
    k1, k2 = jax.random.split(k0)
    data = jax.random.normal(k1, 250) * 2 + 5

    def rpr(k):
        return jnp.exp(jax.random.uniform(k, 2, minval=-3, maxval=3))

    def rmod(k, th):
        return jax.random.normal(k, 250) * th[1] + th[0]

    def sum_stats(dat):
        return jnp.array([jnp.mean(dat), jnp.std(dat)])

    ssd = sum_stats(data)

    def dist(ss):
        diff = ss - ssd
        return jnp.sqrt(jnp.sum(diff * diff))

    def rdis(k, th):
        return dist(sum_stats(rmod(k, th)))

    p, d = jsmfsb.abc_run(k2, 100, rpr, rdis)
    assert len(p) == 100
    assert len(d) == 100


def test_abcsmcstep():
    k0 = jax.random.key(42)
    k1, k2 = jax.random.split(k0)
    data = jax.random.normal(k1, 250) * 2 + 5

    def rpr(k):
        return jnp.exp(jax.random.uniform(k, 2, minval=-3, maxval=3))

    def rmod(k, th):
        return jax.random.normal(k, 250) * jnp.exp(th[1]) + jnp.exp(th[0])

    def sum_stats(dat):
        return jnp.array([jnp.mean(dat), jnp.std(dat)])

    ssd = sum_stats(data)

    def dist(ss):
        diff = ss - ssd
        return jnp.sqrt(jnp.sum(diff * diff))

    def rdis(k, th):
        return dist(sum_stats(rmod(k, th)))

    n = 100
    keys = jax.random.split(k2, n)
    samples = jax.lax.map(rpr, keys)
    th, lw = jsmfsb.abc_smc_step(
        k0,
        lambda x: jnp.log(jnp.sum(((x < 3) & (x > -3)) / 6)),
        samples,
        jnp.zeros(n) + jnp.log(1 / n),
        rdis,
        lambda k, x: jax.random.normal(k) * 0.1 + x,
        lambda x, y: jnp.sum(jsp.stats.norm.logpdf(y, x, 0.1)),
        10,
    )
    assert th.shape == (n, 2)
    assert len(lw) == n


def test_abcsmc():
    k0 = jax.random.key(42)
    k1, k2 = jax.random.split(k0)
    data = jax.random.normal(k1, 250) * 2 + 5

    def rpr(k):
        return jnp.exp(jax.random.uniform(k, 2, minval=-3, maxval=3))

    def rmod(k, th):
        return jax.random.normal(k, 250) * jnp.exp(th[1]) + jnp.exp(th[0])

    def sum_stats(dat):
        return jnp.array([jnp.mean(dat), jnp.std(dat)])

    ssd = sum_stats(data)

    def dist(ss):
        diff = ss - ssd
        return jnp.sqrt(jnp.sum(diff * diff))

    def rdis(k, th):
        return dist(sum_stats(rmod(k, th)))

    n = 100
    post = jsmfsb.abc_smc(
        k2,
        n,
        rpr,
        lambda x: jnp.sum(jnp.log(((x < 3) & (x > -3)) / 6)),
        rdis,
        lambda k, x: jax.random.normal(k) * 0.1 + x,
        lambda x, y: jnp.sum(jsp.stats.norm.logpdf(y, x, 0.1)),
    )
    assert post.shape == (n, 2)


# eof
