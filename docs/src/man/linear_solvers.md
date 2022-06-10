# Linear solvers

We suppose that the KKT system has been assembled previously
into a given `AbstractKKTSystem`. Then, it remains to compute
the Newton step by solving the KKT system for a given
right-hand-side (given as a `AbstractKKTVector`).
That's exactly the role of the linear solver.

If we do not assume any structure, the KKT system writes
```math
K x = b
```
with $$K$$ the KKT matrix and $$b$$ the current right-hand-side.
MadNLP provides a suite of specialized linear solvers to solve
the linear system.

## Inertia detection
If the matrix $$K$$ has negative eigenvalues, we have no guarantee
that the solution of the KKT system is a descent direction with regards
to the original nonlinear problem. That's the reason why most of the linear
solvers compute the inertia
of the linear system when factorizing the matrix $$K$$ when employed inside
an interior-point algorithm. The inertia counts the number of positive,
negative and zero eigenvalues in the matrix. If the inertia does not
meet a given criteria, then the matrix $$K$$ is regularized by adding
a multiple of the identity to it: $$K_r = K + \alpha I$$.

!!! note
    We recall that the inertia of a matrix $$K$$ is given as
    a triplet $$(n,m,p)$$, with $$n$$ the number of positive eigenvalues,
    $$m$$ the number of negative eigenvalues and $$p$$ the number of
    zero eigenvalues.


## Factorization algorithm
In nonlinear programming, it is common
to employ a Bunch-Kaufman factorization (or LDL factorization)
to factorize the matrix $$K$$, as this algorithm returns the inertia
of the matrix directly as a result of the factorization.

!!! note
    When MadNLP runs in inertia-free mode, the algorithm
    does not require to compute the inertia when factorizing
    the matrix $$K$$. In that case, MadNLP can use a classical
    LU or QR factorization to solve the linear system $$Kx = b$$.


## Solving a KKT system with MadNLP

We suppose available a `AbstractKKTSystem` `kkt`, properly assembled.
We can query the assembled matrix $$K$$ as
```julia
K = MadNLP.get_kkt(kkt)

```
Then, we can create a new linear solver instance as
```julia
linear_solver = MadNLPUmfpack.Solver(K)

```
The instance `linear_solver` does not copy the matrix $$K$$ and
instead keep a reference to it.
```julia
linear_solver.tril === K
```
That way every time we re-assemble the matrix $$K$$ in `kkt`,
the values are directly updated in `linear_solver`.

Then, to recompute the factorization inside `linear_solver`,
one simply as to call:
```julia
MadNLP.factorize!(linear_solver)

```
Once the factorization computed, computing the backsolve
for a right-hand-side `b` simply amounts to
```julia
MadNLP.solve!(linear_solver, b)
```
The values of `b` being modified inplace to store the solution $$x$$ of the linear
system $$Kx =b $$.

