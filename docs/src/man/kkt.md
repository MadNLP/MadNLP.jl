```@meta
CurrentModule = MadNLP
```
```@setup kkt_example
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
\end{pmatrix}}^{K_{1}}
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
The matrix $$K_1$$ is unsymmetric, but we can obtain an equivalent symmetric
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

In MadNLP, the matrix $$K_1$$ is encoded as an [`AbstractUnreducedKKTSystem`](@ref)
and the matrix $$K_2$$ is encoded as an [`AbstractReducedKKTSystem`](@ref).


## Assembling a KKT system, step by step

We note that both $$K_1$$ and $$K_2$$ depend on the Hessian
of the Lagrangian $$W$$, the Jacobian $$A$$ and the
diagonal matrices $$\Sigma_x = X^{1}V$$ and $$\Sigma_s = S^{-1}W$$.
Hence, we have to update the KKT system at each iteration
of the interior-point algorithm.

In what follows, we illustrate the inner working of any `AbstractKKTSystem`
by using the KKT system used by default inside MadNLP: [`SparseKKTSystem`](@ref).

### Initializing a KKT system

The size of the KKT system depends directly on the problem's characteristics
(number of variables, number of of equality and inequality constraints).
A [`SparseKKTSystem`](@ref) stores the Hessian and the Jacobian in sparse
(COO) format. Depending on how we parameterize the system,
it can output either a sparse matrix or a dense matrix (according to the linear solver
we are employing under the hood).

For instance, we can parameterize a sparse KKT system as
```@example kkt_example
T = Float64
VT = Vector{T}
MT = SparseMatrixCSC{T, Int}
kkt = MadNLP.SparseKKTSystem{T, VT, MT}(nlp)
kkt.aug_com

```
and a dense KKT system as
```@example kkt_example
T = Float64
VT = Vector{T}
MT = Matrix{T}
kkt = MadNLP.SparseKKTSystem{T, VT, MT}(nlp)
kkt.aug_com

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
x = NLPModels.get_x0(nlp)
l = zeros(m)

NLPModels.hess_coord!(nlp, x, l, hess_values)

```
Eventually, a post-processing step can be applied to refresh all the values internally:
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

!!! note
    On the contrary to `compress_hessian!`, `compress_jacobian!` is not
    doing nothing by default. Instead, the post-processing step scales the values
    of the Jacobian row by row, applying the scaling of the constraints
    as computed initially by MadNLP.

#### Updating the values of the diagonal matrices
Once the Hessian and the Jacobian updated, it remains
to udpate the values of the diagonal matrix $$\Sigma_x = X^{-1} V$$
and $$\Sigma_s = S^{-1} W$$. In the KKT's interface, this amounts
to call the [`regularize_diagonal!`](@ref) function:
```@example kkt_example
pr_values = ones(n + m)
du_values = zeros(m)

MadNLP.regularize_diagonal!(kkt, pr_values, du_values)

```
where `pr_values` stores the diagonal values for the primal
terms (accounting both for $$\Sigma_x$$ and $$\Sigma_s$$) and `du_values`
stores the diagonal values for the dual terms (mostly used during
feasibility restoration).

### Assembling the KKT matrix
Once the values updated, one can assemble the resulting KKT matrix.
This translates to
```@example kkt_example
MadNLP.build_kkt!(kkt)
kkt_matrix
```
By doing so, the values stored inside `kkt` will be transferred
to the KKT matrix `kkt_matrix` (as returned by the function [`get_kkt`](@ref)).

In details, a [`SparseKKTSystem`](@ref) stores internally the KKT system's values using
a sparse COO format. When [`build_kkt!`](@ref) is called, the sparse COO matrix
is transferred to `SparseMatrixCSC` (if `MT = SparseMatrixCSC`) or a `Matrix`
(if `MT = Matrix`), or any format suitable for factorizing the KKT system
inside a [linear solver](linear_solvers.md).

