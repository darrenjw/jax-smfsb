# inference.py
# Code relating to Chapters 10 and 11

import jax
import jax.numpy as jnp
from jax import jit


def metropolisHastings(key, init, logLik, rprop,
                       ldprop=lambda n, o: 1, ldprior=lambda x: 1,
                       iters=10000, thin=10, verb=True):
    """Run a Metropolis-Hastings MCMC algorithm for the parameters of a
    Bayesian posterior distribution

    Run a Metropolis-Hastings MCMC algorithm for the parameters of a
    Bayesian posterior distribution. Note that the algorithm carries
    over the old likelihood from the previous iteration, making it
    suitable for problems with expensive likelihoods, and also for
    "exact approximate" pseudo-marginal or particle marginal MH
    algorithms.
    
    Parameters
    ----------
    key: JAX random number key
      A key to seed the simulation.
    init : vector
      A parameter vector with which to initialise the MCMC algorithm.
    logLik : function
      A function which takes a parameter (such as `init`) as its
      only required argument and returns the log-likelihood of the
      data. Note that it is fine for this to return the log of an
      unbiased estimate of the likelihood, in which case the
      algorithm will be an "exact approximate" pseudo-marginal MH
      algorithm.
    rprop : stochastic function
      A function which takes a random key and a current parameter
      as its two required arguments and returns a single sample
      from a proposal distribution.
    ldprop : function
      A function which takes a new and old parameter as its first
      two required arguments and returns the log density of the
      new value conditional on the old. Defaults to a flat function which
      causes this term to drop out of the acceptance probability.
      It is fine to use the default for _any_ _symmetric_ proposal,
      since the term will also drop out for any symmetric proposal.
    ldprior : function
      A function which take a parameter as its only required
      argument and returns the log density of the parameter value
      under the prior. Defaults to a flat function which causes this 
      term to drop out of the acceptance probability. People often use 
      a flat prior when they are trying to be "uninformative" or
      "objective", but this is slightly naive. In particular, what
      is "flat" is clearly dependent on the parametrisation of the
      model.
    iters : int
      The number of MCMC iterations required (_after_ thinning).
    thin : int
      The required thinning factor. eg. only store every `thin`
      iterations.
    verb : boolean
      Boolean indicating whether some progress information should
      be printed to the console. Defaults to `True`.

    Returns
    -------
    A matrix with rows representing samples from the posterior
    distribution.

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax.scipy as jsp
    >>> k0 = jax.random.key(42)
    >>> k1, k2 = jax.random.split(k0)
    >>> data = jax.random.normal(k1, 250)*2 + 5
    >>> llik = lambda x: jnp.sum(jsp.stats.norm.logpdf(data, x[0], x[1]))
    >>> prop = lambda k, x: jax.random.normal(k, 2)*0.1 + x
    >>> jsmfsb.metropolisHastings(k2, jnp.array([1.0,1.0]), llik, prop)
    """
    def step(s, k):
        [x, ll] = s
        k1, k2 = jax.random.split(k)
        prop = rprop(k1, x)
        llprop = logLik(prop)
        a = (llprop - ll + ldprior(prop) -
             ldprior(x) + ldprop(x, prop) - ldprop(prop, x))
        accept = (jnp.log(jax.random.uniform(k2)) < a)
        s = [jnp.where(accept, prop, x), jnp.where(accept, llprop, ll)]
        return s, s
    def itera(s, k):
        if (verb):
            jax.debug.print("{s}", s=s)
        keys = jax.random.split(k, thin)
        _, states = jax.lax.scan(step, s, keys)
        final = [states[0][thin-1], states[1][thin-1]]
        return final, final
    keys = jax.random.split(key, iters)
    _, states = jax.lax.scan(itera, [init, -jnp.inf], keys)
    return states[0]
    





# eof

