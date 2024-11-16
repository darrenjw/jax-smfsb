The jax-smfsb tutorial
----------------------

This tutorial assumes that the package has already been installed, following the instructions in the `package readme <https://pypi.org/project/jsmfsb/>`__.

We begin with non-spatial stochastic simulation.

Non-spatial simulation
----------------------

Using a model built-in to the library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First, see how to simulate a built-in (Lotka-Volterra predator-prey)
model:

.. code:: python

   import jax
   import jsmfsb

   lvmod = jsmfsb.models.lv()
   step = lvmod.step_gillespie()
   k0 = jax.random.key(42)
   out = jsmfsb.sim_time_series(k0, lvmod.m, 0, 30, 0.1, step)
   assert(out.shape == (300, 2))

Here we used the ``lv`` model. Other built-in models include ``id`` (immigration-death), ``bd`` (birth-death), ``dimer`` (dimerisation kinetics), ``mm`` (Michaelis-Menten enzyme kinetics) and ``sir`` (SIR epdiemic model). The models are of class ``Spn`` (stochastic Petri net), the main data type used in the package. Note the use of the ``step_gillespie`` method, defined on all ``Spn`` models, which returns a function for simulating from the transition kernel of the model, using the Gillespie algorithm. This function can be used with the ``sim_time_series`` function for simulating model trajectories on a regular time grid. Note that all stochastic simulation functions in this package take a `JAX random number key <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ as their first argument. Alternative simulation algorithms include ``step_poisson`` (Poisson time-stepping), ``step_cle`` (Euler-Maruyama simulation from the associated chemical Langevin equation) and ``step_euler`` (Euler simulation from the continuous deterministic approximation to the model).

If you have ``matplotlib`` installed (``pip install matplotlib``), then
you can also plot the results with:

.. code:: python

   import matplotlib.pyplot as plt
   fig, axis = plt.subplots()
   for i in range(2):
       axis.plot(range(out.shape[0]), out[:,i])

   axis.legend(lvmod.n)
   fig.savefig("lv.pdf")

Standard python docstring documentation is available. Usage information
can be obtained from the python REPL with commands like
``help(jsmfsb.Spn)``, ``help(jsmfsb.Spn.step_gillespie)`` or
``help(jsmfsb.sim_time_series)``. This documentation is also available
on `ReadTheDocs <https://jax-smfsb.readthedocs.io/>`__. The API
documentation contains minimal usage examples.

Creating and simulating a model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Next, let’s create and simulate our own (SIR epidemic) model by
specifying a stochastic Petri net ``Spn`` object explicitly. We must provide species and reaction names, stoichiometry matrices, reaction rates and initial conditions. This time we use approximate Poisson simulation rather than exact simulation via the Gillespie algorithm.

.. code:: python

   import jax.numpy as jnp
   sir = jsmfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
       [[1,1,0], [0,1,0]], [[0,2,0], [0,0,1]],
       lambda x, t: jnp.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
       [197.0, 3, 0])
   step_sir = sir.step_poisson()
   sample = jsmfsb.sim_sample(k0, 500, sir.m, 0, 20, step_sir)
   fig, axis = plt.subplots()
   axis.hist(sample[:,1], 30)
   axis.set_title("Infected at time 20")
   plt.savefig("sIr.pdf")

Here, rather than simulating a time series trajectory, we instead simulate a sample of 500 values from the transition kernel at time 20 using ``sim_sample``.


Reading and parsing models in SBML and SBML-shorthand
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that you can read in `SBML <https://sbml.org/>`__ or `SBML-shorthand <https://pypi.org/project/sbmlsh/>`__ models that have been
designed for discrete stochastic simulation into a stochastic Petri net
directly. To read and parse an SBML model, use

.. code:: python

   m = jsmfsb.file_to_spn("myModel.xml")

Note that if you are working with SBML models in Python using
`libsbml <https://pypi.org/project/python-libsbml/>`__, then there is
also a function ``model_to_spn`` which takes a libsbml model object.

To read and parse an SBML-shorthand model, use

.. code:: python

   m = jsmfsb.mod_to_spn("myModel.mod")

There is also a function ``shorthand_to_spn`` which expects a python
string containing a shorthand model. This is convenient for embedding
shorthand models inside python scripts, and is particularly convenient
when working with things like Jupyter notebooks. Below follows a
complete session to illustrate the idea by creating and simulating a
realisation from a discrete stochastic SEIR model.

.. code:: python

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

A `collection of appropriate
models <https://github.com/darrenjw/smfsb/tree/master/models>`__ is
associated with the book.

Spatial simulation
------------------

In addition to methods such as ``step_gillespie`` and ``step_cle`` for well-mixed simulation, ``Spn`` objects also have methods such as ``step_gillespie_1d`` and ``step_cle_2d`` for 1d and 2d spatially explicit simulation of reaction-diffusion processes on a regular grid. 

1d simulation
~~~~~~~~~~~~~


2d simulation
~~~~~~~~~~~~~



Bayesian parameter inference
----------------------------

ABC
~~~


ABC-SMC
~~~~~~~


PMMH particle MCMC
~~~~~~~~~~~~~~~~~~




Converting from the ``smfsb`` python package
--------------------------------------------

The API for this package is very similar to that of the ``smfsb``
package. The main difference is that non-deterministic (random)
functions have an extra argument (typically the first argument) that
corresponds to a JAX random number key. See the `relevant
section <https://jax.readthedocs.io/en/latest/random-numbers.html>`__ of
the JAX documentation for further information regarding random numbers
in JAX code.


Further information
-------------------

For further information, see the `demo
directory <https://github.com/darrenjw/jax-smfsb/tree/main/demos>`__ and
the `API
documentation <https://jax-smfsb.readthedocs.io/en/latest/index.html>`__.
Within the demos directory, see
`shbuild.py <https://github.com/darrenjw/jax-smfsb/tree/main/demos/shbuild.py>`__
for an example of how to specify a (SEIR epidemic) model using
SBML-shorthand and
`step_cle_2df.py <https://github.com/darrenjw/jax-smfsb/tree/main/demos/step_cle_2df.py>`__
for a 2-d reaction-diffusion simulation. For parameter inference (from
time course data), see
`abc-cal.py <https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc-cal.py>`__
for ABC inference,
`abc_smc.py <https://github.com/darrenjw/jax-smfsb/tree/main/demos/abc_smc.py>`__
for ABC-SMC inference and
`pmmh.py <https://github.com/darrenjw/jax-smfsb/tree/main/demos/pmmh.py>`__
for particle marginal Metropolis-Hastings MCMC-based inference. There
are many other demos besides these.




