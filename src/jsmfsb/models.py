# models.py

import jax.numpy as jnp

from spn import Spn


# some pre-defined models


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
    >>> import smfsb
    >>> lv = smfsb.models.lv()
    >>> step = lv.stepGillespie()
    >>> smfsb.simTs(lv.m, 0, 50, 0.1, step)
    """
    return Spn(["Prey", "Predator"], ["Prey rep", "Inter", "Pred death"],
               [[1,0],[1,1],[0,1]], [[2,0],[0,2],[0,0]],
               lambda x, t: jnp.array([th[0]*x[0], th[1]*x[0]*x[1], th[2]*x[1]]),
               [50,100])







# eof

