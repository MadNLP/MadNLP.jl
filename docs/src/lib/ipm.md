```@meta
CurrentModule = MadNLP
```

# MadNLP solver

MadNLP takes as input a nonlinear program encoded as
a `AbstractNLPModel` and solve it using interior-point.
The main entry point is the function `madnlp`:
```@docs
madnlp
MadNLPExecutionStats
```

In detail, the function [`madnlp`](@ref) builds a `MadNLPSolver` storing all
the required structures in the solution algorithm. Once the
`MadNLPSolver` instantiated, the function `solve!` is applied to solve the
nonlinear program with MadNLP's interior-point algorithm.

```@docs
MadNLPSolver

```
