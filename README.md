# JAX-SMfSB (jsmfsb)

[![](https://readthedocs.org/projects/jax-smfsb/badge/)](https://jax-smfsb.readthedocs.io/en/latest/index.html) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## SMfSB code in Python+JAX

Python code relating to the book [Stochastic Modelling for Systems Biology, third edition](https://github.com/darrenjw/smfsb/).

There is a regular Python+Numpy package on PyPI, [smfsb](https://pypi.org/project/smfsb/), which has complete coverage of the book. If you are new to the book and/or this codebase, that might be a simpler place to start.

*This* package covers all of the *core simulation and inference algorithms* from the book, including the parsing of SBML and SBML-shorthand models. These core algorithms will run very fast, using [JAX](https://jax.readthedocs.io/). Computationally intensive algorithms will typically run between 50 and 150 times faster than they would using the regular `smfsb` package, even without a GPU (but YMMV). You must install JAX (which is system dependent), before attempting to install this package. See the [JAX documentation](https://jax.readthedocs.io/en/latest/installation.html) for details, but for a CPU-only installation, it should be as simple as `pip install jax`.

Once you have JAX installed and working correctly, you can install this package with:
```bash
pip install jsmfsb
```
To upgrade already installed package:
```bash
pip install --upgrade jsmfsb
```

**Note** that a number of breaking syntax changes (more pythonic names) were introduced in version 1.1.0. If you upgrade to a version >= 1.1.0 from a version prior to 1.1.0 you will have to update syntax to the new style.


You can test that your installation is working by entering the following at a python prompt:

```python
import jax
import jsmfsb
```
If these both return silently, you are probably good to go.

## Documentation

Note that **the book**, and its associated [github repo](https://github.com/darrenjw/smfsb) is the main source of documentation for this library. The code in the book is in R, but the code in this library is supposed to mirror the R code, but in Python.

For an introduction to this library, see the **[jax-smfsb tutorial](https://jax-smfsb.readthedocs.io/en/latest/source/tutorial.html)**. 

## Converting from the `smfsb` python package

The API for this package is very similar to that of the `smfsb` package. The main difference is that non-deterministic (random) functions have an extra argument (typically the first argument) that corresponds to a JAX random number key. See the [relevant section](https://jax.readthedocs.io/en/latest/random-numbers.html) of the JAX documentation for further information regarding random numbers in JAX code.

## Further information

For further information, see the [demo directory](https://github.com/darrenjw/jax-smfsb/tree/main/demos) and the [API documentation](https://jax-smfsb.readthedocs.io/en/latest/index.html). Within the demos directory, see [shbuild.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/shbuild.py) for an example of how to specify a (SEIR epidemic) model using SBML-shorthand and [sbml-params.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/sbml-params.py) for an example of how to modify the parameters of a model parsed from SBML (or SBML-shorthand). Also see [step_cle_2df.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/step_cle_2df.py) for a 2-d reaction-diffusion simulation. For parameter inference (from time course data), see [abc-cal.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc-cal.py) for ABC inference, [abc_smc.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc_smc.py) for ABC-SMC inference and [pmmh.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/pmmh.py) for particle marginal Metropolis-Hastings MCMC-based inference. There are many other demos besides these.

You can view this package on [GitHub](https://github.com/darrenjw/jax-smfsb) or [PyPI](https://pypi.org/project/jsmfsb/).

### Contributing

If you have problems with this software, please start an [Issue](https://github.com/darrenjw/jax-smfsb/issues) or a [Discussion](https://github.com/darrenjw/jax-smfsb/discussions). Pull requests containing bug fixes are welcome.


**Copyright (C) 2024 Darren J Wilkinson**


