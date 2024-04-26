# models.py
# some pre-defined models

import jax.numpy as jnp
from jsmfsb import Spn


def bd(th=[1, 1.1]):
    """Create a birth-death model

    Create and return a Spn object representing a discrete stochastic
    birth-death model.

    Parameters
    ----------
    th: array
        array of length 2 containing the birth and death rates

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> bd = jsmfsb.models.bd()
    >>> step = bd.stepGillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.simTs(k, bd.m, 0, 50, 0.1, step)
    """
    return Spn(["X"], ["Birth","Death"], [[1],[1]], [[2],[0]],
               lambda x, t: jnp.array([th[0]*x[0], th[1]*x[0]]),
               [100])



def lv(th=[1, 0.005, 0.6]):
    """Create a Lotka-Volterra model

    Create and return a Spn object representing a discrete stochastic
    Lotka-Volterra model.
    
    Parameters
    ----------
    th: array
        array of length 3 containing the rates of the three governing reactions,
        prey reproduction, predator-prey interaction, and predator death

    Returns
    -------
    Spn model object with rates `th`

    Examples
    --------
    >>> import jsmfsb
    >>> import jax
    >>> lv = jsmfsb.models.lv()
    >>> step = lv.stepGillespie()
    >>> k = jax.random.key(42)
    >>> jsmfsb.simTs(k, lv.m, 0, 50, 0.1, step)
    """
    return Spn(["Prey", "Predator"], ["Prey rep", "Inter", "Pred death"],
               [[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
               lambda x, t: jnp.array([th[0]*x[0], th[1]*x[0]*x[1], th[2]*x[1]]),
               [50,100])







# eof

