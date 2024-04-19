# JAX-SMfSB

## SMfSB code in Python+JAX

Python code relating to the book *Stochastic Modelling for Systems Biology, third edition*.
There is a regular Python+Numpy package on PyPI, [smfsb](https://pypi.org/project/smfsb/), which has complete coverage of the book. If you are new to the book and/or this codebase, that is probably a better place to start.
**This package is currently a WIP**, but in any case will only ever cover the *core simulation and inference algorithms* from the book. However, these core algorithms will run very fast, using [JAX](https://jax.readthedocs.io/). You must install JAX (which is system dependent), before attempting to install this package. See the JAX documentation for details.

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

For further information, see the [demos](https://github.com/darrenjw/jax-smfsb/tree/main/demos) and the [API documentation](https://jax-smfsb.readthedocs.io/en/latest/index.html).

You can view this package on [GitHub](https://github.com/darrenjw/jax-smfsb) or [PyPI](https://pypi.org/project/jsmfsb/).




**Copyright (C) 2024 Darren J Wilkinson**


