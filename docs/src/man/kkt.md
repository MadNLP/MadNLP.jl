```@meta
CurrentModule = MadNLP
```
```@setup kkt_example
using LinearAlgebra
using SparseArrays
using NLPModels
using MadNLP
using MadNLPTests
nlp = MadNLPTests.HS15Model()

```

# KKT systems

KKT systems are linear system with a special KKT structure.
MadNLP uses a special structure [`AbstractKKTSystem`](@ref) to represent internally
the KKT system. The [`AbstractKKTSystem`](@ref) fulfills two goals:

1. Store the values of the Hessian of the Lagrangian and of the Jacobian.
2. Assemble the corresponding KKT matrix $$K$$.


## A brief look at the math
We recall that at each iteration the interior-point
algorithm aims at solving the following relaxed KKT equations
($$\mu$$ playing the role of a homotopy parameter) with a Newton method:
```math
F_\mu(x, s, y, v, w) = 0
```
with
```math
F_\mu(x, s, y, v, w) =
\left\{
\begin{aligned}
    & \nabla f(x) + A^\top y + \nu + w & (F_1) \\
    & - y - w  & (F_2) \\
    & g(x) - s &  (F_3) \\
    & X v - \mu e_n & (F_4) \\
    & S w - \mu e_m & (F_5)
\end{aligned}
\right.
```

The Newton step associated to the KKT equations writes
```math
\overline{
\begin{pmatrix}
 W & 0 & A^\top & - I & 0 \\
 0 & 0 & -I & 0 & -I \\
 A & -I & 0& 0 & 0 \\
 V & 0 & 0 & X & 0 \\
 0 & W & 0 & 0 & S
\end{pmatrix}}^{K_{3}}
\begin{pmatrix}
    \Delta x \\
    \Delta s \\
    \Delta y \\
    \Delta v \\
    \Delta w
\end{pmatrix}
= -
\begin{pmatrix}
    F_1 \\ F_2 \\ F_3 \\ F_4 \\ F_5
\end{pmatrix}
```
The matrix $$K_3$$ is unsymmetric, but we can obtain an equivalent symmetric
system by eliminating the two last rows:
```math
\overline{
\begin{pmatrix}
 W + \Sigma_x & 0 & A^\top \\
 0 & \Sigma_s & -I \\
 A & -I & 0
\end{pmatrix}}^{K_{2}}
\begin{pmatrix}
    \Delta x \\
    \Delta s \\
    \Delta y
\end{pmatrix}
= -
\begin{pmatrix}
    F_1 + X^{-1} F_4 \\ F_2 + S^{-1} F_5 \\ F_3
\end{pmatrix}
```
with $$\Sigma_x = X^{-1} v$$ and $$\Sigma_s = S^{-1} w$$.
The matrix $$K_2$$, symmetric, has a structure more favorable
for a direct linear solver.

In MadNLP, the matrix $$K_3$$ is encoded as an [`AbstractUnreducedKKTSystem`](@ref)
and the matrix $$K_2$$ is encoded as an [`AbstractReducedKKTSystem`](@ref).


## Assembling a KKT system, step by step

We note that both $$K_3$$ and $$K_2$$ depend on the Hessian
of the Lagrangian $$W$$, the Jacobian $$A$$ and the
diagonal matrices $$\Sigma_x = X^{1}V$$ and $$\Sigma_s = S^{-1}W$$.
Hence, we have to update the KKT system at each iteration
of the interior-point algorithm.

By default, MadNLP stores the KKT system as a [`SparseKKTSystem`](@ref).
The KKT system takes as input a [`SparseCallback`](@ref) wrapping
a given `NLPModel` `nlp`. We instantiate the callback `cb` with
the function [`create_callback`](@ref):
```@example kkt_example
cb = MadNLP.create_callback(
    MadNLP.SparseCallback,
    nlp,
)

```

### Initializing a KKT system

The size of the KKT system depends directly on the problem's characteristics
(number of variables, number of of equality and inequality constraints).
A [`SparseKKTSystem`](@ref) stores the Hessian and the Jacobian in sparse
(COO) format. The KKT matrix can be factorized using either a
dense or a sparse linear solvers. Here we use the solver provided
in Lapack:
```@example kkt_example
linear_solver = LapackCPUSolver
```

We can instantiate a `SparseKKTSystem` using
the function [`create_kkt_system`](@ref):
```@example kkt_example
kkt = MadNLP.create_kkt_system(
    MadNLP.SparseKKTSystem,
    cb,
    linear_solver,
)

```

Once the KKT system built, one has to initialize it
to use it inside the interior-point algorithm:
```@example kkt_example
MadNLP.initialize!(kkt);

```

The user can query the KKT matrix inside `kkt`, simply as
```@example kkt_example
kkt_matrix = MadNLP.get_kkt(kkt)
```
This returns a reference to the KKT matrix stores internally
inside `kkt`. Each time the matrix is assembled inside `kkt`,
`kkt_matrix` is updated automatically.


### Updating a KKT system
We suppose now we want to refresh the values stored in the KKT system.


#### Updating the values of the Hessian
The Hessian part of the KKT system can be queried as
```@example kkt_example
hess_values = MadNLP.get_hessian(kkt)

```
For a `SparseKKTSystem`, `hess_values` is a `Vector{Float64}` storing
the nonzero values of the Hessian.
Then, one can update the vector `hess_values` by using NLPModels.jl:
```@example kkt_example
n = NLPModels.get_nvar(nlp)
m = NLPModels.get_ncon(nlp)
x = NLPModels.get_x0(nlp) # primal variables
l = zeros(m) # dual variables

NLPModels.hess_coord!(nlp, x, l, hess_values)

```
Eventually, a post-processing step is applied to refresh all the values internally:
```@example kkt_example
MadNLP.compress_hessian!(kkt)

```
!!! note
    By default, the function [`compress_hessian!`](@ref) does nothing.
    But it can be required for very specific use-case, for instance building internally a
    Schur complement matrix.


#### Updating the values of the Jacobian
We proceed exaclty the same way to update the values in the Jacobian.
One queries the Jacobian values in the KKT system as
```@example kkt_example
jac_values = MadNLP.get_jacobian(kkt)

```
We can refresh the values with NLPModels as
```@example kkt_example
NLPModels.jac_coord!(nlp, x, jac_values)

```
And then applies a post-processing step as
```@example kkt_example
MadNLP.compress_jacobian!(kkt)

```

#### Updating the values of the diagonal matrices
Once the Hessian and the Jacobian updated, the algorithm
can apply primal and dual regularization terms on the diagonal
of the KKT system, to improve the numerical behavior in the linear solver.
This operation is implemented inside the [`regularize_diagonal!`](@ref) function:
```@example kkt_example
pr_value = 1.0
du_value = 0.0

MadNLP.regularize_diagonal!(kkt, pr_value, du_value)

```

### Assembling the KKT matrix
Once the values updated, one can assemble the resulting KKT matrix.
This translates to
```@example kkt_example
MadNLP.build_kkt!(kkt)
```
By doing so, the values stored inside `kkt` will be transferred
to the KKT matrix `kkt_matrix` (as returned by the function [`get_kkt`](@ref)):
```@example kkt_example
kkt_matrix
```

Internally, a [`SparseKKTSystem`](@ref) stores the KKT system in
a sparse COO format. When [`build_kkt!`](@ref) is called, the sparse COO matrix
is transferred to `SparseMatrixCSC` if the linear solver is sparse,
or alternatively to a `Matrix` if the linear solver is dense.

!!! note
    The KKT system stores only the lower-triangular part of the KKT system,
    as it is symmetric.


## Solution of the KKT system
Now the KKT system is assembled in a matrix ``K`` (here stored in `kkt_matrix`), we want
to solve a linear system ``K x = b``, for instance to evaluate the
next descent direction. To do so, we use the linear solver stored
internally inside `kkt` (here an instance of `LapackCPUSolver`).

We start by factorizing the KKT matrix ``K``:
```@example kkt_example
MadNLP.factorize!(kkt.linear_solver)

```
By default, MadNLP uses a LBL factorization to decompose the symmetric
indefinite KKT matrix.

Once the KKT matrix has been factorized, we can compute the solution of the linear
system with a backsolve. The function takes as input a [`AbstractKKTVector`](@ref),
an object used to do algebraic manipulation with a [`AbstractKKTSystem`](@ref).
We start by instantiating two [`UnreducedKKTVector`](@ref) (encoding respectively
the right-hand-side and the solution):
```@example kkt_example
b = MadNLP.UnreducedKKTVector(kkt)
fill!(MadNLP.full(b), 1.0)
x = copy(b)

```
The right-hand-side encodes a vector of 1:
```@example kkt_example
MadNLP.full(b)
```
We solve the system ``K x = b`` using the [`solve!`](@ref) function:
```@example kkt_example
MadNLP.solve!(kkt, x)
MadNLP.full(x)
```
We verify that the solution is correct by multiplying it on the left
with the KKT system, using `mul!`:
```@example kkt_example
mul!(b, kkt, x) # overwrite b!
MadNLP.full(b)
```
We recover a vector filled with `1`, which was the initial right-hand-side.

