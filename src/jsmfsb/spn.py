#!/usr/bin/env python3
# spn.py

import numpy as np
import jax
from jax import jit
import jax.numpy as jnp
import jax.lax as jl

# Spn class definition, including methods


class Spn:
    """Class for stochastic Petri net models."""

    def __init__(self, n, t, pre, post, h, m):
        """Constructor method for Spn objects

        Create a Spn object for representing a stochastic Petri net model that
        can be simulated using a variety of algorithms.

        Parameters
        ----------
        n : list of strings
            Names of the species/tokens in the model
        t : list of strings
            Names of the reactions/transitions in the model
        pre : matrix
            Matrix representing the LHS stoichiometries
        post: matrix
            Matrix representing the RHS stoichiometries
        h: function
            A function to compute the rates of the reactions from the current state and time of
            the system. The function should return a numpy array of rates.
        m: list of integers
            The intial state/marking of the model/net

        Returns
        -------
        A object of class Spn.

        Examples
        --------
        >>> import jsmfsb
        >>> import jax
        >>> import jax.numpy as jnp
        >>> sir = jsmfsb.Spn(["S", "I", "R"], ["S->I", "I->R"],
              [[1,1,0],[0,1,0]], [[0,2,0],[0,0,1]],
              lambda x, t: jnp.array([0.3*x[0]*x[1]/200, 0.1*x[1]]),
              [197, 3, 0])
        >>> stepSir = sir.step_gillespie()
        >>> jsmfsb.sim_sample(jax.random.key(42), 10, sir.m, 0, 20, stepSir)
        """
        self.n = n  # species names
        self.t = t  # reaction names
        self.pre = jnp.array(pre).astype(jnp.float32)
        self.post = jnp.array(post).astype(jnp.float32)
        self.h = h  # hazard function
        self.m = jnp.array(m).astype(jnp.float32)  # initial marking

    def __str__(self):
        """A very simple string representation of the Spn object, mainly for debugging."""
        return "n: {}\n t: {}\npre: {}\npost: {}\nh: {}\nm: {}".format(
            str(self.n),
            str(self.t),
            str(self.pre),
            str(self.post),
            str(self.h),
            str(self.m),
        )

    def step_gillespie(self, min_haz=1e-10, max_haz=1e07):
        """Create a function for advancing the state of a SPN by using the
        Gillespie algorithm

        This method returns a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `sim_time_series`) for simulating realisations of SPN models.

        Parameters
        ----------
        min_haz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        max_haz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has interface `function(key, x0, t0, deltat)`, where `key` is an
        unused JAX random key, `x0` and `t0` represent the initial state
        and time, and `deltat` represents the amount of time by which the
        process should be advanced. The function closure returns a vector
        representing the simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.step_gillespie()
        >>> stepLv(jax.random.key(42), lv.m, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        @jit
        def advance(state):
            key, xo, x, t = state
            h = self.h(x, t)
            h0 = jnp.sum(h)
            key, k1, k2 = jax.random.split(key, 3)
            t = jnp.where(h0 > max_haz, 1e90, t)
            t = jnp.where(h0 < min_haz, 1e90, t + jax.random.exponential(k1) / h0)
            j = jax.random.choice(k2, v, p=h / h0)
            xn = jnp.add(x, sto[:, j])
            return (key, x, xn, t)

        @jit
        def step(key, x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            key, x, xn, t = jl.while_loop(
                lambda state: state[3] < termt, advance, (key, x, x, t)
            )
            return x

        return step

    def step_poisson(self, dt=0.01):
        """Create a function for advancing the state of an SPN by using a
        simple approximate Poisson time stepping method

        This method returns a function for advancing the state of an SPN
        model using a simple approximate Poisson time stepping method. The
        resulting function (closure) can be used in conjunction with other
        functions (such as ‘sim_time_series’) for simulating realisations of SPN
        models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a Poisson time stepping method with step size
        ‘dt’. The function closure has interface
        ‘function(key, x0, t0, deltat)’, where ‘x0’ and ‘t0’ represent the
        initial state and time, and ‘deltat’ represents the amount of time
        by which the process should be advanced. The function closure
        returns a vector representing the simulated state of the system at
        the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.step_poisson(0.001)
        >>> k = jax.random.key(42)
        >>> stepLv(k, lv.m, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        @jit
        def advance(state):
            key, x, t = state
            key, k1 = jax.random.split(key)
            h = self.h(x, t)
            r = jax.random.poisson(k1, h * dt)
            x = jnp.add(x, sto.dot(r))
            # TODO: sort out negative values
            # x = jnp.where(x < 0, -x, x)
            t = t + dt
            return (key, x, t)

        @jit
        def step(key, x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            key, x, t = jl.while_loop(
                lambda state: state[2] < termt, advance, (key, x, t)
            )
            return x

        return step

    def step_euler(self, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        continuous deterministic Euler integration method

        This method returns a function for advancing the state of an SPN
        model using a simple continuous deterministic Euler integration
        method. The resulting function (closure) can be used in
        conjunction with other functions (such as ‘sim_time_series’) for simulating
        realisations of SPN models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using an Euler method with step size ‘dt’. The
        function closure has interface ‘function(x0, t0, deltat)’, where
        ‘x0’ and ‘t0’ represent the initial state and time, and ‘deltat’
        represents the amount of time by which the process should be
        advanced. The function closure returns a vector representing the
        simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.step_euler(0.001)
        >>> k = jax.random.key(42)
        >>> stepLv(k, lv.m, 0, 1)
        """
        sto = (self.post - self.pre).T

        @jit
        def advance(state):
            key, x, t = state
            key, k1 = jax.random.split(key)
            h = self.h(x, t)
            x = jnp.add(x, sto.dot(h * dt))
            x = jnp.where(x < 0, -x, x)
            t = t + dt
            return (key, x, t)

        @jit
        def step(key, x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            key, x, t = jl.while_loop(
                lambda state: state[2] < termt, advance, (key, x, t)
            )
            return x

        return step

    def step_cle(self, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama integration method for the associated CLE

        This method returns a function for advancing the state of an SPN
        model using a simple Euler-Maruyama integration method
        method for the chemical Langevin equation form of the model.The
        resulting function (closure) can be used in
        conjunction with other functions (such as `sim_time_series`) for simulating
        realisations of SPN models.

        Parameters
        ----------
        dt : float
            The time step for the time-stepping integration method. Defaults to 0.01.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using an Euler-Maruyama method with step size ‘dt’. The
        function closure has interface ‘function(key, x0, t0, deltat)’, where
        ‘x0’ and ‘t0’ represent the initial state and time, and ‘deltat’
        represents the amount of time by which the process should be
        advanced. The function closure returns a vector representing the
        simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> lv = jsmfsb.models.lv()
        >>> stepLv = lv.step_cle(0.001)
        >>> stepLv(jax.random.key(42), lv.m, 0, 1)
        """
        sto = (self.post - self.pre).T
        v = sto.shape[1]
        sdt = np.sqrt(dt)

        @jit
        def advance(state):
            key, x, t = state
            key, k1 = jax.random.split(key)
            h = self.h(x, t)
            dw = jax.random.normal(k1, [v]) * sdt
            x = jnp.add(x, sto.dot(h * dt + jnp.sqrt(h) * dw))
            x = jnp.where(x < 0, -x, x)
            t = t + dt
            return (key, x, t)

        @jit
        def step(key, x0, t0, deltat):
            x = x0
            t = t0
            termt = t0 + deltat
            key, x, t = jl.while_loop(
                lambda state: state[2] < termt, advance, (key, x, t)
            )
            return x

        return step

    # spatial simulation functions, from chapter 9

    def step_gillespie_1d(self, d, min_haz=1e-10, max_haz=1e07):
        """Create a function for advancing the state of an SPN by using the
        Gillespie algorithm on a 1D regular grid

        This method creates a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `sim_time_series_1d`) for simulating realisations of SPN models in space and
        time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore twice this value (as it can leave to
          the left or the right).
        min_haz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        max_haz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has arguments `key`, `x0`, `t0`, `deltat`, where `key` is a JAX
        random key, `x0` is a matrix with rows corresponding to species
        and columns corresponding to voxels, representing the initial
        condition, `t0` represent the initial state and time, and
        `deltat` represents the amount of time by which the process should
        be advanced. The function closure returns a matrix representing
        the simulated state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> import jax.numpy as jnp
        >>> lv = jsmfsb.models.lv()
        >>> stepLv1d = lv.step_gillespie_1d(jnp.array([0.6, 0.6]))
        >>> N = 20
        >>> x0 = jnp.zeros((2,N))
        >>> x0 = x0.at[:,int(N/2)].set(lv.m)
        >>> k0 = jax.random.key(42)
        >>> stepLv1d(k0, x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        @jit
        def advance(state):
            key, xo, x, t = state
            key, k1, k2, k3 = jax.random.split(key, 4)
            hr = jnp.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
            hrs = jnp.apply_along_axis(jnp.sum, 0, hr)
            hrss = hrs.sum()
            hd = jnp.apply_along_axis(lambda xi: xi * d * 2, 0, x)
            hds = jnp.apply_along_axis(jnp.sum, 0, hd)
            hdss = hds.sum()
            h0 = hrss + hdss
            t = jnp.where(h0 > max_haz, 1e90, t)
            t = jnp.where(h0 < min_haz, 1e90, t + jax.random.exponential(k1) / h0)

            def diffuse(key, x):
                n = x.shape[1]
                k1, k2, k3 = jax.random.split(key, 3)
                j = jax.random.choice(k1, n, p=hds / hdss)  # pick a box
                i = jax.random.choice(k2, u, p=hd[:, j] / hds[j])  # pick species
                x = x.at[i, j].set(x[i, j] - 1)  # decrement chosen box
                ind = jnp.where(jax.random.uniform(k3) < 0.5, j - 1, j + 1)
                ind = jnp.where(ind < 0, n - 1, ind)
                ind = jnp.where(ind > n - 1, 0, ind)
                x = x.at[i, ind].set(x[i, ind] + 1)  # increment new box
                return x

            def react(key, x):
                n = x.shape[1]
                k1, k2 = jax.random.split(key, 2)
                j = jax.random.choice(k1, n, p=hrs / hrss)  # pick a box
                i = jax.random.choice(k2, v, p=hr[:, j] / hrs[j])  # pick a reaction
                x = x.at[:, j].set(jnp.add(x[:, j], sto[:, i]))
                return x

            xn = jnp.where(
                jax.random.uniform(k2) * h0 < hdss, diffuse(k3, x), react(k3, x)
            )
            return (key, x, xn, t)

        @jit
        def step(key, x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            key, x, xn, t = jl.while_loop(
                lambda state: state[3] < termt, advance, (key, x, x, t)
            )
            return x

        return step

    def step_gillespie_2d(self, d, min_haz=1e-10, max_haz=1e07):
        """Create a function for advancing the state of an SPN by using the
        Gillespie algorithm on a 2D regular grid

        This method creates a function for advancing the state of an SPN
        model using the Gillespie algorithm. The resulting function
        (closure) can be used in conjunction with other functions (such as
        `sim_time_series_2d`) for simulating realisations of SPN models in space and
        time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore four times this value (as it can leave in
          one of 4 directions).
        min_haz : float
          Minimum hazard to consider before assuming 0. Defaults to 1e-10.
        max_haz : float
          Maximum hazard to consider before assuming an explosion and
          bailing out. Defaults to 1e07.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using the Gillespie algorithm. The function closure
        has arguments `key`, `x0`, `t0`, `deltat`, where `key` is a JAX random
        key, `x0` is a 3d array with dimensions corresponding to species
        then two spatial dimensions, representing the initial condition,
        `t0` represents the time of the initial state, and `deltat`
        represents the amount of time by which the process should be advanced.
        The function closure returns an array representing the simulated
        state of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> import jax.numpy as jnp
        >>> lv = jsmfsb.models.lv()
        >>> stepLv2d = lv.step_gillespie_2d(jnp.array([0.6, 0.6]))
        >>> N = 20
        >>> x0 = jnp.zeros((2, N, N))
        >>> x0 = x0.at[:, int(N/2), int(N/2)].set(lv.m)
        >>> k0 = jax.random.key(42)
        >>> stepLv2d(k0, x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape

        @jit
        def advance(state):
            key, xo, x, t = state
            key, k1, k2, k3 = jax.random.split(key, 4)
            hr = jnp.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
            hrs = jnp.sum(hr, axis=(0))
            hrss = hrs.sum()
            hd = jnp.apply_along_axis(lambda xi: xi * d * 4, 0, x)
            hds = jnp.sum(hd, axis=(0))
            hdss = hds.sum()
            h0 = hrss + hdss
            t = jnp.where(h0 > max_haz, 1e90, t)
            t = jnp.where(h0 < min_haz, 1e90, t + jax.random.exponential(k1) / h0)

            def diffuse(key, x):
                uu, m, n = x.shape
                k1, k2, k3 = jax.random.split(key, 3)
                r = jax.random.choice(k1, m * n, p=hds.flatten() / hdss)  # pick a box
                i, j = divmod(r, n)
                k = jax.random.choice(k2, u, p=hd[:, i, j] / hds[i, j])  # pick species
                x = x.at[k, i, j].set(x[k, i, j] - 1)  # decrement chosen box
                un = jax.random.uniform(k3)
                ind = jnp.where(
                    un < 0.25,
                    jnp.array([i, j - 1]),
                    jnp.where(
                        un < 0.5,
                        jnp.array([i, j + 1]),
                        jnp.where(
                            un < 0.75, jnp.array([i - 1, j]), jnp.array([i + 1, j])
                        ),
                    ),
                )
                i = ind[0]
                j = ind[1]
                i = jnp.where(i < 0, m - 1, i)
                i = jnp.where(i > m - 1, 0, i)
                j = jnp.where(j < 0, n - 1, j)
                j = jnp.where(j > n - 1, 0, j)
                x = x.at[k, i, j].set(x[k, i, j] + 1)  # increment new box
                return x

            def react(key, x):
                uu, m, n = x.shape
                k1, k2 = jax.random.split(key, 2)
                r = jax.random.choice(k1, m * n, p=hrs.flatten() / hrss)  # pick a box
                i, j = divmod(r, n)
                k = jax.random.choice(
                    k2, v, p=hr[:, i, j] / hrs[i, j]
                )  # pick a reaction
                x = x.at[:, i, j].set(jnp.add(x[:, i, j], sto[:, k]))
                return x

            xn = jnp.where(
                jax.random.uniform(k2) * h0 < hdss, diffuse(k3, x), react(k3, x)
            )
            return (key, x, xn, t)

        @jit
        def step(key, x0, t0, deltat):
            t = t0
            x = x0
            termt = t0 + deltat
            key, x, xn, t = jl.while_loop(
                lambda state: state[3] < termt, advance, (key, x, x, t)
            )
            return x

        return step

    def step_cle_1d(self, d, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama discretisation of the CLE on a 1D regular grid

        This method creates a function for advancing the state of an SPN
        model using a simple Euler-Maruyama discretisation of the CLE on a
        1D regular grid. The resulting function (closure) can be used in
        conjunction with other functions (such as `sim_time_series_1d`) for
        simulating realisations of SPN models in space and time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore twice this value (as it can leave to
          the left or the right).
        dt : float
          Time step for the Euler-Maruyama discretisation.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a simple Euler-Maruyama algorithm. The function
        closure has parameters `key`, `x0`, `t0`, `deltat`, where `key` is a
        JAX random number key, `x0` is
        a matrix with rows corresponding to species and columns
        corresponding to voxels, representing the initial condition, `t0`
        represents the initial state and time, and `deltat` represents the
        amount of time by which the process should be advanced. The
        function closure returns a matrix representing the simulated state
        of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> import jax.numpy as jnp
        >>> lv = jsmfsb.models.lv()
        >>> stepLv1d = lv.step_cle_1d(jnp.array([0.6,0.6]))
        >>> N = 20
        >>> x0 = jnp.zeros((2,N))
        >>> x0 = x0.at[:,int(N/2)].set(lv.m)
        >>> k0 = jax.random.key(42)
        >>> stepLv1d(k0, x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape
        sdt = np.sqrt(dt)

        def forward(m):
            return jnp.roll(m, -1, axis=1)

        def back(m):
            return jnp.roll(m, +1, axis=1)

        def laplacian(m):
            return forward(m) + back(m) - 2 * m

        def rectify(m):
            return jnp.where(m < 0, 0, m)

        def diffuse(key, m):
            n = m.shape[1]
            noise = jax.random.normal(key, (u, n)) * sdt
            m = (
                m
                + (jnp.diag(d) @ laplacian(m)) * dt
                + jnp.diag(jnp.sqrt(d))
                @ (
                    jnp.sqrt(m + forward(m)) * noise
                    - jnp.sqrt(m + back(m)) * back(noise)
                )
            )
            m = rectify(m)
            return m

        def step(key, x0, t0, deltat):
            n = x0.shape[1]
            tt = int(deltat // dt) + 1
            keys = jax.random.split(key, tt)

            def advance(state, key):
                k1, k2 = jax.random.split(key)
                x, t = state
                t = t + dt
                x = diffuse(k1, x)
                hr = jnp.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
                dwt = jax.random.normal(k2, (v, n)) * sdt
                x = x + sto @ (hr * dt + jnp.diag(jnp.sqrt(hr)) @ dwt)
                x = rectify(x)
                return (x, t), x

            _, out = jl.scan(advance, (x0, t0), keys)
            return out[tt - 1]

        step = jit(step, static_argnums=(3,))
        return step

    def step_cle_2d(self, d, dt=0.01):
        """Create a function for advancing the state of an SPN by using a simple
        Euler-Maruyama discretisation of the CLE on a 2D regular grid

        This method creates a function for advancing the state of an SPN
        model using a simple Euler-Maruyama discretisation of the CLE on a
        2D regular grid. The resulting function (closure) can be used in
        conjunction with other functions (such as `sim_time_series_2d`) for
        simulating realisations of SPN models in space and time.

        Parameters
        ----------
        d : array
          A vector of diffusion coefficients - one coefficient for each
          reacting species, in order. The coefficient is the reaction
          rate for a reaction for a molecule moving into an adjacent
          compartment. The hazard for a given molecule leaving the
          compartment is therefore four times this value (as it can leave
          in one of 4 directions).
        dt : float
          Time step for the Euler-Maruyama discretisation.

        Returns
        -------
        A function which can be used to advance the state of the SPN
        model by using a simple Euler-Maruyama algorithm. The function
        closure has parameters `key`, `x0`, `t0`, `deltat`, where `x0` is
        a 3d array with indices species, then rows and columns
        corresponding to voxels, representing the initial condition, `t0`
        represents the initial state and time, and `deltat` represents the
        amount of time by which the process should be advanced. The
        function closure returns a matrix representing the simulated state
        of the system at the new time.

        Examples
        --------
        >>> import jsmfsb.models
        >>> import jax
        >>> import jax.numpy as jnp
        >>> lv = jsmfsb.models.lv()
        >>> stepLv2d = lv.step_cle_2d(jnp.array([0.6,0.6]))
        >>> M = 15
        >>> N = 20
        >>> x0 = jnp.zeros((2,M,N))
        >>> x0 = x0.at[:,int(M/2),int(N/2)].set(lv.m)
        >>> k0 = jax.random.key(42)
        >>> stepLv2d(k0, x0, 0, 1)
        """
        sto = (self.post - self.pre).T
        u, v = sto.shape
        sdt = np.sqrt(dt)

        def left(a):
            return jnp.roll(a, -1, axis=1)

        def right(a):
            return jnp.roll(a, +1, axis=1)

        def up(a):
            return jnp.roll(a, -1, axis=2)

        def down(a):
            return jnp.roll(a, +1, axis=2)

        def laplacian(a):
            return left(a) + right(a) + up(a) + down(a) - 4 * a

        def rectify(a):
            return jnp.where(a < 0, 0, a)

        def diffuse(key, a):
            uu, m, n = a.shape
            k1, k2 = jax.random.split(key)
            dwt = jax.random.normal(k1, (u, m, n)) * sdt
            dwts = jax.random.normal(k2, (u, m, n)) * sdt
            a = (
                a
                + (jnp.apply_along_axis(lambda xi: xi * d, 0, laplacian(a))) * dt
                + jnp.apply_along_axis(
                    lambda xi: xi * jnp.sqrt(d),
                    0,
                    (
                        jnp.sqrt(a + left(a)) * dwt
                        - jnp.sqrt(a + right(a)) * right(dwt)
                        + jnp.sqrt(a + up(a)) * dwts
                        - jnp.sqrt(a + down(a)) * down(dwts)
                    ),
                )
            )
            a = rectify(a)
            return a

        def react(si):
            si = si.reshape(2, v)
            hri = si[0]
            dwti = si[1]
            return sto @ (hri * dt + jnp.sqrt(hri) * dwti)

        def step(key, x0, t0, deltat):
            uu, m, n = x0.shape
            tt = int(deltat // dt) + 1
            keys = jax.random.split(key, tt)

            def advance(state, key):
                k1, k2 = jax.random.split(key)
                x, t = state
                t = t + dt
                x = diffuse(k1, x)
                hr = jnp.apply_along_axis(lambda xi: self.h(xi, t), 0, x)
                dwt = jax.random.normal(k2, (v, m, n)) * sdt
                # TODO: this would be neater (and maybe faster) with "vmap"
                stacked = jnp.stack((hr, dwt))  # (2, v, m, n)
                x = x + jnp.apply_along_axis(react, 0, stacked.reshape(2 * v, m, n))
                x = rectify(x)
                return (x, t), x

            _, out = jl.scan(advance, (x0, t0), keys)
            return out[tt - 1]

        step = jit(step, static_argnums=(3,))
        return step


# eof
