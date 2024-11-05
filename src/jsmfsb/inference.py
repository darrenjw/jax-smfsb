# inference.py
# Code relating to Chapters 10 and 11

import jax
import jax.numpy as jnp
from jax import jit


# MCMC functions


def metropolis_hastings(
    key,
    init,
    log_lik,
    rprop,
    ldprop=lambda n, o: 1,
    ldprior=lambda x: 1,
    iters=10000,
    thin=10,
    verb=True,
):
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
    log_lik : (stochastic) function
      A function which takes two arguments: a JAX random key and
      a parameter (the same type as `init`) as its
      second argument. It should return the log-likelihood of the
      data. Note that it is fine for this to return the log of an
      unbiased estimate of the likelihood, in which case the
      algorithm will be an "exact approximate" pseudo-marginal MH
      algorithm. This is the reason why the function should accept
      a JAX random key. In the "vanilla" case, where the log-likelihood
      is deterministic, the function should simply ignore the key that
      is passed in.
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
    >>> llik = lambda k, x: jnp.sum(jsp.stats.norm.logpdf(data, x[0], x[1]))
    >>> prop = lambda k, x: jax.random.normal(k, 2)*0.1 + x
    >>> jsmfsb.metropolis_hastings(k2, jnp.array([1.0,1.0]), llik, prop)
    """

    def step(s, k):
        [x, ll] = s
        k1, k2, k3 = jax.random.split(k, 3)
        prop = rprop(k1, x)
        llprop = log_lik(k2, prop)
        a = llprop - ll + ldprior(prop) - ldprior(x) + ldprop(x, prop) - ldprop(prop, x)
        accept = jnp.log(jax.random.uniform(k3)) < a
        s = [jnp.where(accept, prop, x), jnp.where(accept, llprop, ll)]
        return s, s

    def itera(s, k):
        if verb:
            jax.debug.print("{s}", s=s)
        keys = jax.random.split(k, thin)
        _, states = jax.lax.scan(step, s, keys)
        final = [states[0][thin - 1], states[1][thin - 1]]
        return final, final

    keys = jax.random.split(key, iters)
    _, states = jax.lax.scan(itera, [init, -jnp.inf], keys)
    return states[0]


def pf_marginal_ll(n, sim_x0, t0, step_fun, data_ll, data, debug=False):
    """Create a function for computing the log of an unbiased estimate of
    marginal likelihood of a time course data set

    Create a function for computing the log of an unbiased estimate of
    marginal likelihood of a time course data set using a simple
    bootstrap particle filter.

    Parameters
    ----------
    n :  int
      An integer representing the number of particles to use in the
      particle filter.
    sim_x0 : function
      A function with arguments `key`, `t0` and `th`, where ‘t0’ is a time
      at which to simulate from an initial distribution for the state of the
      particle filter and `th` is a vector of parameters. The return value
      should be a state vector randomly sampled from the prior distribution.
      The function therefore represents a prior distribution on the initial
      state of the Markov process.
    t0 : float
      The time corresponding to the starting point of the Markov
      process. Can be no bigger than the smallest observation time.
    step_fun : function
      A function for advancing the state of the Markov process, with
      arguments `key`, `x`, `t0`, `deltat` and `th`, with `th` representing a
      vector of parameters.
    data_ll : function
      A function with arguments `x`, `t`, `y`, `th`,
      where `x` and `t` represent the true state and time of the
      process, `y` is the observed data, and `th` is a parameter vector.
      The return value should be the log of the likelihood of the observation. The
      function therefore represents the observation model.
    data : matrix
      A matrix with first column an increasing set of times. The remaining
      columns represent the observed values of `y` at those times.

    Returns
    -------
    A function with arguments `key` and `th`, representing a parameter vector, which
    evaluates to the log of the particle filters unbiased estimate of the
    marginal likelihood of the data (for parameter `th`).

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> import jax.scipy as jsp
    >>> import jsmfsb
    >>> def obsll(x, t, y, th):
    >>>     return jnp.sum(jsp.stats.norm.logpdf(y-x, scale=10))
    >>>
    >>> def simX(key, t0, th):
    >>>     k1, k2 = jax.random.split(key)
    >>>     return jnp.array([jax.random.poisson(k1, 50),
    >>>              jax.random.poisson(k2, 100)]).astype(jnp.float32)
    >>>
    >>> def step(key, x, t, dt, th):
    >>>     sf = jsmfsb.models.lv(th).step_gillespie()
    >>>     return sf(key, x, t, dt)
    >>>
    >>> mll = jsmfsb.pf_marginal_ll(80, simX, 0, step, obsll, jsmfsb.data.lv_noise_10)
    >>> k0 = jax.random.key(42)
    >>> mll(k0, jnp.array([1, 0.005, 0.6]))
    >>> mll(k0, jnp.array([2, 0.005, 0.6]))
    """
    no = data.shape[1]
    times = jnp.concatenate((jnp.array([t0]), data[:, 0]))
    deltas = jnp.diff(times)
    obs = data[:, 1:no]
    if debug:
        print(data.shape)
        print(times[range(5)])
        print(deltas[range(5)])
        print(len(deltas))
        print(obs[range(5), :])

    @jit
    def go(key, th):
        key, k1 = jax.random.split(key)
        keys = jax.random.split(k1, n)
        xmat = jax.lax.map(lambda k: sim_x0(k, t0, th), keys)
        sh = xmat.shape
        if debug:
            print(sh)
            print(xmat[range(5), :])

        def advance(state, key):
            [i, xmat, ll] = state
            assert xmat.shape == sh
            key, k1, k2 = jax.random.split(key, 3)
            keys = jax.random.split(k1, n)

            def prop(k, x):
                return step_fun(k, x, times[i], deltas[i], th)

            xmat = jax.vmap(prop)(keys, xmat)
            lw = jnp.apply_along_axis(
                lambda x: data_ll(x, times[i + 1], obs[i,], th), 1, xmat
            )
            m = jnp.max(lw)
            sw = jnp.exp(lw - m)
            ssw = jnp.sum(sw)
            rows = jax.random.choice(k2, n, shape=(n,), p=sw / ssw)
            state = [i + 1, xmat[rows, :], ll + m + jnp.log(ssw / n)]
            return state, state

        keys = jax.random.split(key, len(deltas))
        _, states = jax.lax.scan(advance, [0, xmat, 0.0], keys)
        return states[2][len(deltas)]

    return go


# ABC functions


def abc_run(key, n, rprior, rdist, batch_size=None, verb=False):
    """Run a set of simulations initialised with parameters sampled from a
    given prior distribution, and compute statistics required for an ABC
    analaysis

    Run a set of simulations initialised with parameters sampled from
    a given prior distribution, and compute statistics required for an
    ABC analaysis. Typically used to calculate "distances" of
    simulated synthetic data from observed data.

    Parameters
    ----------
    key: JAX random number key
      Key to initialise the ABC simulation.
    n : int
      An integer representing the number of simulations to run.
    rprior : function
      A function with one argument, a JAX random key, generating
      a single parameter (vector) from a prior distribution.
    rdist : function
      A function with two arguments, a JAX random key, and a
      parameter (vector). It returns the required statistic of
      interest. This will typically be computed by first using
      the parameter to run a
      forward model, then computing required summary statistics,
      then computing a distance. See the example for details.
    batch_size: int
      batch_size to use in call to jax.lax.map for parallelisation.
      Defaults to None.
    verb : boolean
      Print progress information to console? Defaults to False.

    Returns
    -------
    A tuple with first component a matrix of parameters (in rows)
    and second component a vector of corresponding distances.

    Examples
    --------
    >>> import jsmfsb
    >>> import jax.numpy as jnp
    >>> import jax.scipy as jsp
    >>> k0 = jax.random.key(42)
    >>> k1, k2 = jax.random.split(k0)
    >>> data = jax.random.normal(k1, 250)*2 + 5
    >>> def rpr(k):
    >>>   return jnp.exp(jax.random.uniform(k, 2, minval=-3, maxval=3))
    >>>
    >>> def rmod(k, th):
    >>>   return jax.random.normal(k, 250)*th[1] + th[0]
    >>>
    >>> def sumStats(dat):
    >>>   return jnp.array([jnp.mean(dat), jnp.std(dat)])
    >>>
    >>> ssd = sumStats(data)
    >>> def dist(ss):
    >>>   diff = ss - ssd
    >>>   return jnp.sqrt(jnp.sum(diff*diff))
    >>>
    >>> def rdis(k, th):
    >>>   return dist(sumStats(rmod(k, th)))
    >>>
    >>> smfsb.abc_run(k2, 100, rpr, rdis)
    """

    @jit
    def pair(k):
        k1, k2 = jax.random.split(k)
        p = rprior(k1)
        d = rdist(k2, p)
        if verb:
            jax.debug.print("{p}, {d}", p=p, d=d)
        return (p, d)

    keys = jax.random.split(key, n)
    sims = jax.lax.map(pair, keys, batch_size=batch_size)
    return sims


# ABC-SMC functions


def abc_smc_step(
    key, dprior, prior_sample, prior_lw, rdist, rperturb, dperturb, factor
):
    """Carry out one step of an ABC-SMC algorithm

    Not meant to be directly called by users. See abc_smc.
    """
    k1, k2, k3 = jax.random.split(key, 3)
    n = prior_sample.shape[0]
    mx = jnp.max(prior_lw)
    rw = jnp.exp(prior_lw - mx)
    prior_ind = jax.random.choice(k1, n, shape=(n * factor,), p=rw / jnp.sum(rw))
    prior = prior_sample[prior_ind, :]
    keys = jax.random.split(k2, len(prior_ind))
    prop = jax.vmap(rperturb)(keys, prior)
    keys2 = jax.random.split(k3, len(prior_ind))
    dist = jax.vmap(rdist)(keys2, prop)  # this is typically the slow step
    q_cut = jnp.nanquantile(dist, 1 / factor)
    new = prop[dist < q_cut, :]

    def log_weight(th):
        terms = prior_lw + jnp.apply_along_axis(
            lambda x: dperturb(th, x), 1, prior_sample
        )
        mt = jnp.max(terms)
        denom = mt + jnp.log(jnp.sum(jnp.exp(terms - mt)))
        return dprior(th) - denom

    lw = jnp.apply_along_axis(log_weight, 1, new)
    mx = jnp.max(lw)
    rw = jnp.exp(lw - mx)
    nlw = jnp.log(rw / jnp.sum(rw))
    return new, nlw


def abc_smc(
    key,
    n,
    rprior,
    dprior,
    rdist,
    rperturb,
    dperturb,
    factor=10,
    steps=15,
    verb=False,
    debug=False,
):
    """Run an ABC-SMC algorithm for infering the parameters of a forward model

    Run an ABC-SMC algorithm for infering the parameters of a forward
    model. This sequential Monte Carlo algorithm often performs better
    than simple rejection-ABC in practice.

    Parameters
    ----------
    key : JAX random key
      A key to initialise the simulation.
    n : int
      An integer representing the number of simulations to pass on
      at each stage of the SMC algorithm. Note that the TOTAL
      number of forward simulations required by the algorithm will
      be (roughly) 'N*steps*factor'.
    rprior : function
      A function with a single argument, a JAX random key, which generates
      a single parameter (vector) from the prior.
    dprior : function
      A function taking a parameter vector as argumnent and returning
      the log of the prior density.
    rdist : function
      A function with two arguments: a JAX random key and a parameter vector.
      It should return a scalar "distance" representing a measure of how
      good the chosen parameter is. This will typically be computed
      by first using the parameter to run a forward model, then
      computing required summary statistics, then computing a
      distance. See the example for details.
    rperturb : function
      A function with two arguments: a JAX random key and a parameter vector.
      It should return a perturbed parameter from an appropriate kernel.
    dperturb : function
      A function which takes a pair of parameters as its first two
      arguments (new first and old second), and returns the log of the density
      associated with this perturbation kernel.
    factor : int
      At each step of the algorithm, 'N*factor' proposals are
      generated and the best 'N' of these are weighted and passed
      on to the next stage. Note that the effective sample size of
      the parameters passed on to the next step may be (much)
      smaller than 'N', since some of the particles may be assigned
      small (or zero) weight. Defaults to 10.
    steps : int
      The number of steps of the ABC-SMC algorithm. Typically,
      somewhere between 5 and 100 steps seems to be used in
      practice. Defaults to 15.
    verb : boolean
      Boolean indicating whether some progress should be printed to
      the console.

    Returns
    -------
    A matrix with rows representing samples from the approximate posterior
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
    >>> def rpr(k):
    >>>   return jnp.exp(jax.random.uniform(k, 2, minval=-3, maxval=3))
    >>>
    >>> def rmod(k, th):
    >>>   return jax.random.normal(k, 250)*jnp.exp(th[1]) + jnp.exp(th[0])
    >>>
    >>> def sumStats(dat):
    >>>   return jnp.array([jnp.mean(dat), jnp.std(dat)])
    >>>
    >>> ssd = sumStats(data)
    >>> def dist(ss):
    >>>   diff = ss - ssd
    >>>   return jnp.sqrt(jnp.sum(diff*diff))
    >>>
    >>> def rdis(k, th):
    >>>   return dist(sumStats(rmod(k, th)))
    >>>
    >>> jsmfsb.abc_smc(k2, 100, rpr,
    >>>                        lambda x: jnp.sum(jnp.log(((x<3)&(x>-3))/6)),
    >>>                        rdis,
    >>>                        lambda k,x: jax.random.normal(k)*0.1 + x,
    >>>                        lambda x,y: jnp.sum(jsp.stats.norm.logpdf(y, x, 0.1)))
    """
    key, k1 = jax.random.split(key)
    prior_lw = jnp.log(jnp.zeros((n)) + 1 / n)
    keys = jax.random.split(k1, n)
    prior_sample = jax.lax.map(rprior, keys)
    # TODO: worth turning this loop into a "scan"? Maybe not.
    for i in range(steps):
        key, k1 = jax.random.split(key)
        if verb:
            print(steps - i, end=" ", flush=True)
        prior_sample, prior_lw = abc_smc_step(
            k1, dprior, prior_sample, prior_lw, rdist, rperturb, dperturb, factor
        )
        if debug:
            print(prior_sample.shape)
            print(prior_lw.shape)
    if verb:
        print("Done.")
    if debug:
        print(prior_sample.shape)
        print(prior_lw.shape)
    ind = jax.random.choice(key, prior_lw.shape[0], shape=(n,), p=jnp.exp(prior_lw))
    return prior_sample[ind, :]


# eof
