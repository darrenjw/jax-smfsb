#!/usr/bin/env python3

import jsmfsb
import jax
import jax.scipy as jsp
import jax.numpy as jnp
import time

sir_sh = """
@model:3.1.1=SEIR "SEIR Epidemic model"
 s=item, t=second, v=litre, e=item
@compartments
 Pop
@species
 Pop:S=197 s
 Pop:I=3 s
 Pop:R=0 s
@reactions
@r=Infection
 S + I -> 2I
 beta*S*I : beta=0.0015
@r=Removal
 I -> R
 gamma*I : gamma=0.1
"""

sir = jsmfsb.shorthand_to_spn(sir_sh)
step_sir = sir.step_gillespie()
k0 = jax.random.key(42)
out = jsmfsb.sim_time_series(k0, sir.m, 0, 40, 0.05, step_sir)
print("Starting timed run now")
# start timer
start_time = time.time()
out = jsmfsb.sim_sample(k0, 10000, sir.m, 0, 20, step_sir, batch_size=100)
# end timer
end_time = time.time()
elapsed = end_time - start_time
print(f"\n\nElapsed time: {elapsed} seconds\n\n")
print(jnp.apply_along_axis(jnp.mean, 0, out))

# Compare with built-in version
sir = jsmfsb.models.sir()
step_sir = sir.step_gillespie()
out = jsmfsb.sim_time_series(k0, sir.m, 0, 40, 0.05, step_sir)
print("Starting timed run now")
# start timer
start_time = time.time()
out = jsmfsb.sim_sample(k0, 10000, sir.m, 0, 20, step_sir, batch_size=100)
# end timer
end_time = time.time()
elapsed = end_time - start_time
print(f"\n\nElapsed time: {elapsed} seconds\n\n")
print(jnp.apply_along_axis(jnp.mean, 0, out))


# eof
