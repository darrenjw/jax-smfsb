# spatial code from chapter 9
# Note that the actual simulation code is in the Spn object in the spn module

import jax
import jax.numpy as jnp

def simTs1D(key, x0, t0, tt, dt, stepFun, verb=False):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a 1D
    regular spatial grid and regular grid of times using a function
    (closure) for advancing the state of the model, such as created by
    `stepGillespie1D`.

    Parameters
    ----------
    key: JAX random number key
      Initial random number key to seed the simulation.
    x0 : array
      The initial state of the process at time `t0`, a matrix with
      rows corresponding to reacting species and columns
      corresponding to spatial location.
    t0 : float
      The initial time to be associated with the initial state `x0`.
    tt : float
      The terminal time of the simulation.
    dt : float
      The time step of the output. Note that this time step relates
      only to the recorded output, and has no bearing on the
      accuracy of the simulation process.
    stepFun : function
      A function (closure) for advancing the state of the process,
      such as produced by `stepGillespie1D`.
    verb : boolean
      Output progress to the console (this function can be very slow).

    Returns
    -------
    A 3d array representing the simulated process. The dimensions
    are species, space, and time.

    Examples
    --------
    >>> import smfsb.models
    >>> import numpy as np
    >>> lv = smfsb.models.lv()
    >>> stepLv1d = lv.stepGillespie1D(np.array([0.6,0.6]))
    >>> N = 10
    >>> T = 5
    >>> x0 = np.zeros((2,N))
    >>> x0[:,int(N/2)] = lv.m
    >>> smfsb.simTs1D(x0, 0, T, 1, stepLv1d, True)
    """
    N = int((tt - t0)//dt + 1)
    u, n = x0.shape
    arr = jnp.zeros((u, n, N))
    x = x0
    t = t0
    arr = arr.at[:,:,0].set(x)
    # TODO: replace with a scan operation
    for i in range(1, N):
        key, k1 = jax.random.split(key)
        if (verb):
            print(N-i)
        t = t + dt
        x = stepFun(k1, x, t, dt)
        arr = arr.at[:,:,i].set(x)
    return(arr)
    

def simTs2D(x0, t0, tt, dt, stepFun, verb=False):
    """Simulate a model on a regular grid of times, using a function (closure)
    for advancing the state of the model

    This function simulates single realisation of a model on a 2D
    regular spatial grid and regular grid of times using a function
    (closure) for advancing the state of the model, such as created by
    `stepGillespie2D`.

    Parameters
    ----------
    x0 : array
      The initial state of the process at time `t0`, a 3d array with
      dimensions corresponding to reacting species and then two
      corresponding to spatial location.
    t0 : float
      The initial time to be associated with the initial state `x0`.
    tt : float
      The terminal time of the simulation.
    dt : float
      The time step of the output. Note that this time step relates
      only to the recorded output, and has no bearing on the
      accuracy of the simulation process.
    stepFun : function
      A function (closure) for advancing the state of the process,
      such as produced by `stepGillespie2D`.
    verb : boolean
      Output progress to the console (this function can be very slow).

    Returns
    -------
    A 4d array representing the simulated process. The dimensions
    are species, two space, and time.

    Examples
    --------
    >>> import smfsb.models
    >>> import numpy as np
    >>> lv = smfsb.models.lv()
    >>> stepLv2d = lv.stepGillespie2D(np.array([0.6,0.6]))
    >>> M = 10
    >>> N = 15
    >>> T = 5
    >>> x0 = np.zeros((2,M,N))
    >>> x0[:,int(M/2),int(N/2)] = lv.m
    >>> smfsb.simTs2D(x0, 0, T, 1, stepLv2d, True)
    """
    N = int((tt - t0)//dt + 1)
    u, m, n = x0.shape
    arr = np.zeros((u, m, n, N))
    x = x0
    t = t0
    arr[:,:,:,0] = x
    for i in range(1, N):
        if (verb):
            print(N-i)
        t = t + dt
        x = stepFun(x, t, dt)
        arr[:,:,:,i] = x
    return(arr)
    


# eof


