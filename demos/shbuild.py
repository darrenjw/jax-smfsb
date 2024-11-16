#!/usr/bin/env python3
# shbuild.py
# build a model with SBML-shorthand

import jax
import matplotlib.pyplot as plt
import scipy as sp

import jsmfsb

seir_sh = """
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

seir = jsmfsb.shorthand_to_spn(seir_sh)
step_seir = seir.step_gillespie()
k0 = jax.random.key(42)
out = jsmfsb.sim_time_series(k0, seir.m, 0, 40, 0.05, step_seir)

fig, axis = plt.subplots()
for i in range(4):
    axis.plot(range(out.shape[0]), out[:, i])

axis.legend(seir.n)
fig.savefig("shbuild.pdf")

# sim_sample
out = jsmfsb.sim_sample(k0, 10000, seir.m, 0, 10, step_seir)

print(sp.stats.describe(out))
fig, axes = plt.subplots(4, 1)
for i in range(4):
    axes[i].hist(out[:, i], bins=20)
fig.savefig("shbuildH.pdf")

# eof
