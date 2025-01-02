# test_models.py
# Tests to run using pytest

import dsmts
import jax.numpy as jnp

N = 10000


def check_model(file_stem):
    fails = dsmts.test_model(N, file_stem)
    assert jnp.sum(fails) == 0


# List of models to check:

# 001 models


def test_m0001():
    check_model("stochastic/00001/dsmts-001-01")


def test_m0002():
    check_model("stochastic/00002/dsmts-001-02")


# 002 models


def test_m0020():
    check_model("stochastic/00020/dsmts-002-01")


def test_m0021():
    check_model("stochastic/00021/dsmts-002-02")


def test_m0022():
    check_model("stochastic/00022/dsmts-002-03")


# 003 models


def test_m0030():
    check_model("stochastic/00030/dsmts-003-01")


def test_m0031():
    check_model("stochastic/00031/dsmts-003-02")


# eof
