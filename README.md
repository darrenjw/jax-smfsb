# JAX-SMfSB (jsmfsb)

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

## Basic usage

Note that **the book**, and its associated [github repo](https://github.com/darrenjw/smfsb) is the main source of documentation for this library. The code in the book is in R, but the code in this library is supposed to mirror the R code, but in Python.

### Using a model built-in to the library

First, see how to simulate a built-in (Lotka-Volterra predator-prey) model:

```python
import jax
import jsmfsb

lvmod = jsmfsb.models.lv()
step = lvmod.step_gillespie()
k0 = jax.random.key(42)
out = jsmfsb.sim_time_series(k0, lvmod.m, 0, 30, 0.1, step)
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

Standard python docstring documentation is available. Usage information can be obtained from the python REPL with commands like `help(jsmfsb.Spn)`, `help(jsmfsb.Spn.step_gillespie)` or `help(jsmfsb.sim_time_series)`. This documentation is also available on [ReadTheDocs](https://jax-smfsb.readthedocs.io/). The API documentation contains very minimal usage examples. For more interesting examples, see the [demos directory](https://github.com/darrenjw/jax-smfsb/tree/main/demos).

### Creating and simulating a model

Next, let's create and simulate our own (SIR epidemic) model by specifying a stochastic Petri net explicitly.
```python
import jax.numpy as jnp
sir = jsmfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
	[[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
	lambda x, t: jnp.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
	[197.0, 3, 0])
step_sir = sir.step_poisson()
sample = jsmfsb.sim_sample(k0, 500, sir.m, 0, 20, step_sir)
fig, axis = plt.subplots()
axis.hist(sample[:,1], 30)
axis.set_title("Infected at time 20")
plt.savefig("sIr.pdf")
```

### Reading and parsing models in SBML and SBML-shorthand

Note that you can read in SBML or SBML-shorthand models that have been designed for discrete stochastic simulation into a stochastic Petri net directly. To read and parse an SBML model, use
```python
m = jsmfsb.file_to_spn("myModel.xml")
```
Note that if you are working with SBML models in Python using [libsbml](https://pypi.org/project/python-libsbml/), then there is also a function `model_to_spn` which takes a libsbml model object.

To read and parse an SBML-shorthand model, use
```python
m = jsmfsb.mod_to_spn("myModel.mod")
```
There is also a function `shorthand_to_spn` which expects a python string containing a shorthand model. This is convenient for embedding shorthand models inside python scripts, and is particularly convenient when working with things like Jupyter notebooks. Below follows a complete session to illustrate the idea by creating and simulating a realisation from a discrete stochastic SEIR model.
```python
import jax
import jsmfsb
import jax.numpy as jnp

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

import matplotlib.pyplot as plt
fig, axis = plt.subplots()
for i in range(len(seir.m)):
	axis.plot(jnp.arange(0, 40, 0.05), out[:,i])

axis.legend(seir.n)
fig.savefig("seir.pdf")
```

A [collection of appropriate models](https://github.com/darrenjw/smfsb/tree/master/models) is associated with the book.

## Converting from the `smfsb` python package

The API for this package is very similar to that of the `smfsb` package. The main difference is that non-deterministic (random) functions have an extra argument (typically the first argument) that corresponds to a JAX random number key. See the [relevant section](https://jax.readthedocs.io/en/latest/random-numbers.html) of the JAX documentation for further information regarding random numbers in JAX code.

## Further information

For further information, see the [demo directory](https://github.com/darrenjw/jax-smfsb/tree/main/demos) and the [API documentation](https://jax-smfsb.readthedocs.io/en/latest/index.html). Within the demos directory, see [shbuild.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/shbuild.py) for an example of how to specify a (SEIR epidemic) model using SBML-shorthand and [step_cle_2df.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/step_cle_2df.py) for a 2-d reaction-diffusion simulation. For parameter inference (from time course data), see [abc-cal.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc-cal.py) for ABC inference, [abc_smc.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc_smc.py) for ABC-SMC inference and [pmmh.py](https://github.com/darrenjw/jax-smfsb/tree/main/demos/pmmh.py) for particle marginal Metropolis-Hastings MCMC-based inference. There are many other demos besides these.

You can view this package on [GitHub](https://github.com/darrenjw/jax-smfsb) or [PyPI](https://pypi.org/project/jsmfsb/).

### Contributing

If you have problems with this software, please start an [Issue](https://github.com/darrenjw/jax-smfsb/issues) or a [Discussion](https://github.com/darrenjw/jax-smfsb/discussions). Pull requests containing bug fixes are welcome.


**Copyright (C) 2024 Darren J Wilkinson**


