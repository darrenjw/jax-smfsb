#!/usr/bin/env python3
# shbuild.py
# build a model with SBML-shorthand

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.lax as jl

import jsmfsb

seirSH = """
@model:3.1.1=SEIR "SEIR Epidemic model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:S=100 s
 Pop:E=0 s	  
 Pop:I=5 s
 Pop:R=0 s
@reactions
@r=Infection
 S + I -> E + I
 beta*S*I : beta=0.1
@r=Transition
 E -> I
 sigma*E : sigma=0.2
@r=Removal
 I -> R
 gamma*I : gamma=0.5
"""

seir = jsmfsb.sh2Spn(seirSH)
stepSeir = seir.stepGillespie()
k0 = jax.random.key(42)
out = jsmfsb.simTs(k0, seir.m, 0, 40, 0.05, stepSeir)

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(4):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(seir.n)
fig.savefig("shbuild.pdf")

# simSample
out = jsmfsb.simSample(k0, 10000, seir.m, 0, 10, stepSeir)
import scipy as sp
print(sp.stats.describe(out))
fig, axes = plt.subplots(4,1)
for i in range(4):
    axes[i].hist(out[:,i], bins=20)
fig.savefig("shbuildH.pdf")

# eof

