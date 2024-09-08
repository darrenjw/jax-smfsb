# JAX-SMfSB (jsmfsb)

## SMfSB code in Python+JAX

Python code relating to the book [Stochastic Modelling for Systems Biology, third edition](https://github.com/darrenjw/smfsb/).

There is a regular Python+Numpy package on PyPI, [smfsb](https://pypi.org/project/smfsb/), which has complete coverage of the book. If you are new to the book and/or this codebase, that might be a simpler place to start.

*This* package covers all of the *core simulation and inference algorithms* from the book, including the parsing of SBML and SBML-shorthand models. These core algorithms will run very fast, using [JAX](https://jax.readthedocs.io/). Computationally intensive algorithms will typically run between 50 and 150 times faster than they would using the regular `smfsb` package, even without a GPU (but YMMV). You must install JAX (which is system dependent), before attempting to install this package. See the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html) for details, but for a CPU-only installation, it should be as simple as `pip install jax`.

Once you have JAX installed and working correctly, you can install this package with:
```bash
pip install jsmfsb
```

You can test that your installation is working with the following example.
```python
import jax
import jsmfsb

lvmod = jsmfsb.models.lv()
step = lvmod.stepGillespie()
k0 = jax.random.key(42)
out = jsmfsb.simTs(k0, lvmod.m, 0, 30, 0.1, step)
assert(out.shape == (300, 2))
```

If you have `matplotlib` installed (`pip install matplotlib`), then you can also plot the results with:
```python
import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(2):
	axis.plot(range(out.shape[0]), out[:,i])

axis.legend(lvmod.n)
fig.savefig("lv.pdf")
```

The API for this package is very similar to that of the `smfsb` package. The main difference is that non-deterministic (random) functions have an extra argument (typically the first argument) that corresponds to a JAX random number key. See the [relevant section](https://jax.readthedocs.io/en/latest/random-numbers.html) of the JAX documentation for further information regarding random numbers in JAX code.

For further information, see the [demo directory](https://github.com/darrenjw/jax-smfsb/tree/main/demos) and the [API documentation](https://jax-smfsb.readthedocs.io/en/latest/index.html). Within the demos directory, see [shbuild.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/shbuild.py) for an example of how to specify a (SEIR epidemic) model using SBML-shorthand and [stepCLE2Df.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/stepCLE2Df.py) for a 2-d reaction-diffusion simulation. For parameter inference (from time course data), see [abc-cal.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc-cal.py) for ABC inference, [abcSmc.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/abcSmc.py) for ABC-SMC inference and [pmmh.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/pmmh.py) for particle marginal Metropolis-Hastings MCMC-based inference. There are many other demos besides these.

You can view this package on [GitHub](https://github.com/darrenjw/jax-smfsb) or [PyPI](https://pypi.org/project/jsmfsb/).




**Copyright (C) 2024 Darren J Wilkinson**


