# dsmts.py
# Utilities for running the DSMTS against smfsb

import jsmfsb
import jax
import jax.numpy as jnp
import pandas as pd


def test_model(n, file_stem):
    model_file = file_stem + ".mod"
    mean_file = file_stem + "-mean.csv"
    sd_file = file_stem + "-sd.csv"
    mean = pd.read_csv(mean_file).to_numpy()[:, 1:]
    mean = jnp.array(mean).astype(jnp.float32)
    sd = pd.read_csv(sd_file).to_numpy()[:, 1:]
    sd = jnp.array(sd).astype(jnp.float32)
    spn = jsmfsb.mod_to_spn(model_file)
    step = spn.step_gillespie()  # testing the exact simulator
    u = len(spn.n)
    sx0 = jnp.zeros((51, u))
    sxx0 = jnp.zeros((51, u))
    state0 = [sx0, sxx0]

    def update(state, key):
        [sx0, sxx0] = state
        out = jsmfsb.sim_time_series(key, spn.m, 0, 50, 1, step)
        sx = sx0 + out
        si = out - mean
        sxx = sxx0 + (si * si)
        return ([sx, sxx], 0)

    k0 = jax.random.key(42)
    keys = jax.random.split(k0, n)
    state, _ = jax.lax.scan(update, state0, keys)
    [sx, sxx] = state
    sample_mean = sx / n
    z_scores = jnp.sqrt(n) * (sample_mean - mean) / sd
    sts = sxx / n
    y_scores = (sts / (sd * sd) - 1) * jnp.sqrt(n / 2)
    fails = jnp.array([jnp.sum(abs(z_scores) > 3), jnp.sum(abs(y_scores) > 5)])
    if jnp.sum(fails) > 0:
        print(str(fails) + " FAILS for " + file_stem)
        print(sample_mean)
        print(z_scores)
        print(sts)
        print(y_scores)
    return fails


# Run a demo test if run as a script

if __name__ == "__main__":
    print("A demo test run. Use pytest to run the full suite properly.")
    N = 1000
    print(test_model(N, "stochastic/00001/dsmts-001-01"))
    print(test_model(N, "stochastic/00020/dsmts-002-01"))
    print(test_model(N, "stochastic/00030/dsmts-003-01"))
    print("Done.")


# eof
