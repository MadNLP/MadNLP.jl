# MadCorePardiso.jl

Pardiso linear solvers for MadCore:
- `PardisoMKLSolver` — via the freely-available `MKL_jll`.
- `PardisoSolver` — the proprietary Panua PARDISO; point `JULIA_PARDISO` at the
  library folder to enable it.
