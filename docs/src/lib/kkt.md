```@meta
CurrentModule = MadNLP
```

# KKT systems

MadNLP manipulates KKT systems using two abstractions:
an `AbstractKKTSystem` storing the KKT system' matrix
and an `AbstractKKTVector` storing the KKT system's right-hand-side.


## AbstractKKTSystem

```@docs
AbstractKKTSystem

```

MadNLP implements three different types of `AbstractKKTSystem`,
depending how far we reduce the KKT system.

```@docs
AbstractUnreducedKKTSystem
AbstractReducedKKTSystem
AbstractCondensedKKTSystem
```

Each `AbstractKKTSystem` follows the interface described below:
```@docs

num_variables
get_kkt
get_jacobian
get_hessian

initialize!
build_kkt!
compress_hessian!
compress_jacobian!
jtprod!
regularize_diagonal!
set_jacobian_scaling!
is_inertia_correct
is_reduced
nnz_jacobian
```

By default, MadNLP stores a `AbstractReducedKKTSystem` in sparse format,
as implemented by `SparseKKTSystem`:
```@docs
SparseKKTSystem

```
The user has the choice to store the KKT system as a sparse `AbstractUnreducedKKTSystem`:
```@docs
SparseUnreducedKKTSystem
```

MadNLP provides also two structures to store the KKT system
in a dense matrix. Although less efficient than their sparse counterparts,
these two structures allow to store the KKT system efficiently when the
problem is instantiated on the GPU.
```@docs
DenseKKTSystem
DenseCondensedKKTSystem

```


## AbstractKKTVector
Each instance of `AbstractKKTVector` implements
the following interface.

```@docs
AbstractKKTVector

number_primal
number_dual
full
primal
dual
primal_dual
dual_lb
dual_ub

```

By default, MadNLP provides two different `AbstractKKTVector`.

```@docs
ReducedKKTVector
UnreducedKKTVector
```

