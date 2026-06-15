# MadCore.jl

The solver-agnostic core of the MadSuite: KKT systems, the linear-solver
interface (built-in LAPACK + iterative refinement), NLPModels callbacks,
quasi-Newton approximations, and options/logging.

MadCore has minimal dependencies so that linear-solver backends and downstream
solvers (MadNLP, MadNCL, MadIPM, CCopt) can build on the shared infrastructure
**without importing the full MadNLP interior-point solver**. External and
backend-specific solvers live in `lib/*` subpackages (HSL, MUMPS, Pardiso,
SuiteSparse, LDLFactorizations, and the KernelAbstractions/CUDA/AMDGPU GPU
backends).
