#!/usr/bin/env python3
# abc.py
# ABC using Euclidean distance (no summary stats)

import jsmfsb
import jax
import jax.numpy as jnp
import jax.scipy as jsp

print("ABC")

data = jsmfsb.data.LVperfect[:,1:3]

def rpr(k):
  k1, k2, k3 = jax.random.split(k, 3)
  return jnp.exp(jnp.array([jax.random.uniform(k1, minval=-3, maxval=3),
                          jax.random.uniform(k2, minval=-8, maxval=-2),
                          jax.random.uniform(k3, minval=-4, maxval=2)]))

def rmod(k, th):
  return jsmfsb.simTs(k, jnp.array([50.0, 100.0]), 0, 30, 2,
                     jsmfsb.models.lv(th).step_cle(0.1))

def sumStats(dat):
  return dat

ssd = sumStats(data)

def dist(ss):
  diff = ss - ssd
  return jnp.sqrt(jnp.sum(diff*diff))

def rdis(k, th):
  return dist(sumStats(rmod(k, th)))

k0 = jax.random.key(42)
p, d = jsmfsb.abcRun(k0, 1000000, rpr, rdis, batch_size=100000, verb=False)

q = jnp.nanquantile(d, 0.01)
prmat = jnp.vstack(p)
postmat = prmat[d < q,:]
its, var = postmat.shape
print(its, var)

postmat = jnp.log(postmat) # look at posterior on log scale

import matplotlib.pyplot as plt
fig, axes = plt.subplots(2, 3)
axes[0, 0].scatter(postmat[:,0], postmat[:,1], s=0.5)
axes[0, 1].scatter(postmat[:,0], postmat[:,2], s=0.5)
axes[0, 2].scatter(postmat[:,1], postmat[:,2], s=0.5)
axes[1, 0].hist(postmat[:,0], bins=30)
axes[1, 1].hist(postmat[:,1], bins=30)
axes[1, 2].hist(postmat[:,2], bins=30)
fig.savefig("abc.pdf")

print("All done.")


# eof

