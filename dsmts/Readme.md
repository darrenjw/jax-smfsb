# dsmts

## Discrete stochastic models test suite

Test the exact Gillespie algorithm implementation by using the discrete stochastic models test suite (DSMTS), which is now part of the SBML Test Suite. 

* https://sbml.org/software/sbml-test-suite/
* https://github.com/sbmlteam/sbml-test-suite/releases
* https://doi.org/10.1093/bioinformatics/btm566

Just run against a subset of the models, corresponding to the limited range of SBML features supported by this package.

To run, unpack the zip and just run:

pytest

from this directory. Note that "pandas" is required for CSV parsing,
which isn't part of the requirements for this library.

