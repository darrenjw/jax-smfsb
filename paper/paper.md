---
title: 'jax-smfsb: A python package for stochastic systems biology modelling and inference'
tags:
  - Python
  - JAX
  - systems biology
  - stochastic modelling
  - reaction networks
  - SBML
  - Gillespie algorithm
  - Bayesian inference
  - high-performance
  - parallel
  - GPU
  - reaction-diffusion
authors:
  - name: Darren J. Wilkinson
    orchid: 0000-0003-0736-802X
    affiliation: 1
affiliations:
  - name: Department of Mathematical Sciences, University of Durham, UK
  - index: 1
date: 19 September 2024
bibliography: paper.bib
---

# Summary

Many biological processes, and especially molecular biochemical processes, exhibit non-trivial stochasticity in their dynamical behaviour **W09**. The popular textbook *Stochastic modelling for systems biology, third edition* **W18** describes the stochastic approach to modelling and simulation of biochemical processes, and how to do Bayesian inference **GW11** for the parameters of such models using time course data. `jax-smfsb` provides a fast and efficient implementation of all of the algorithms described in **W18**, able to effectively exploit multiple cores and GPUs, leading to performance suitable for the analysis of non-trivial research problems.

# Statement of Need

Although there exist many tools for modelling biological network dynamics using deterministic approaches, typically based on ordinary differential equations (ODEs), there are relatively few flexible software libraries for modelling and simulation of stochastic biochemical networks. There are even fewer libraries for principled (fully Bayesian) inference for the parameters of such networks using data.

In addition to describing the mathematical framework for stochastic modelling, simulation, and inference, **W18** also describes a software implementation of all of the algorithms. The language chosen to illustrate the implementation was R, and the library is available on CRAN as the R package [`smfsb`](https://cran.r-project.org/package=smfsb). While this library is of significant pedagogical value, the overheads of dynamic interpreted languages such as R make it unsuitable for the development of high-performance codes suitable for tackling non-trivial research problems. An implementation in the compiled strongly-typed functional language Scala **CITE**, [`scala-smfsb`](https://github.com/darrenjw/scala-smfsb) partially addresses this issue, but the lack of systems biology students and researchers familiar with Scala has limited the impact of this library. More recently, a Python **CITE** port of the library, [`python-smfsb`](https://github.com/darrenjw/python-smfsb) has been developed, utilising the Python libraries [`numpy`](https://numpy.org/) **CITE** and [`scipy`](https://scipy.org/) **CITE**. This is of significant pedagogical value, since Python has become a more popular programming language for systems biology modelling than R. Nevertheless, the performance of this library is similar to that of the R library, inadequate for serious research problems.

`jax-smfsb` addresses all of the limitations of the previously described implementations. It is essentially a port of `python-smfsb` with `numpy` and `scipy` replaced by [JAX](https://github.com/google/jax) **CITE**. JAX is a state-of-the-art high-performance machine learning framework that turns out to be well-suited to a range of problems in numerical, scientific and statistical computing. JAX is effectively a functional language for differentiable array processing embedded in Python, allowing just-in-time compilation and execution on modern hardware with state-of-the-art performance. In addition to a large number of machine learning libraries based on JAX, a growing ecosystem of libraries for scientific computing is developing; see, for example, [diffrax](https://docs.kidger.site/diffrax/) **CITE**, [jax-md](https://github.com/jax-md/jax-md) **CITE**, [JAX-Fluids](https://github.com/tumaer/JAXFLUIDS) **CITE**. `jax-smfsb` adds to this ecosystem by providing tools for modelling, simulation and Bayesian inference for stochastic (biochemical) network models.

# Features




# References

