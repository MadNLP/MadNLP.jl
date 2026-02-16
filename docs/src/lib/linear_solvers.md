```@meta
CurrentModule = MadNLP
```

# Linear solvers

## Direct linear solvers
Each linear solver employed in MadNLP
implements the following interface.

```@docs
AbstractLinearSolver
introduce
factorize!
solve_linear_system!
is_inertia
inertia
```

## Iterative refinement
MadNLP uses iterative refinement to improve the
accuracy of the solution returned by the linear solver.

```@docs
solve_refine!

```
