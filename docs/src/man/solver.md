
MadNLP is an interior-point solver based on a filter line-search.
We detail here the inner machinery happening at each MadNLP's iteration.

We recall that MadNLP is a primal-dual interior-point method
and starts from an initial primal-dual variables $$(x_0, s_0, y_0)$$.

## What happens at iteration k?

At iteration $$k$$, MadNLP aims at finding a new iterate
$$(x_{k+1}, s_{k+1}, y_{k+1})$$ improving the current
iterate $$(x_{k}, s_{k}, y_{k})$$, in the sense that
the new iterate (i) improves the objective or (ii) decreases the
infeasibility. The exact trade-off between (i) and (ii) is
handled by the filter line-search.

The algorithm follows the steps:

1. Check problem convergence $$E_0(x_k, s_k, y_k) < \varepsilon_{tol}$$
2. If necessary, update the barrier parameter $$\mu_k$$
3. Evaluate the Hessian of the Lagrangian $$W_k$$ and the Jacobian $$A_k$$ with the callbacks
4. Assemble the KKT system and compute the search direction $$d_k$$ by solving the resulting linear system.
5. If necessary, regularize the KKT system to guarantee that $$d_k$$ is a descent direction
6. Run the backtracking line-search and find a step $$\alpha_k$$
7. Define the next iterate as $$x_{k+1} = x_k + \alpha_k d_k^x$$, $$s_{k+1} = s_k + \alpha_k d_k^s$$, $$y_{k+1} = y_k + \alpha_k d_k^y$$.

We detail each step in the following paragraphs.

!!! note
    In general, Step 3 (Hessian and Jacobian evaluations) and Step 4 (solving the KKT system) are the two most numerically demanding steps.

## Step 1: When the algorithm stops?

MadNLP stops once the solution satisfies a specified accuracy $$\varepsilon_{tol}$$
(by default $$10^{-8}$$). MadNLP uses the same stopping criterion
as Ipopt by defining
```math
E_\mu(x_k, s_k; y_k, \nu_k, w_k) =
\max \left\{
\begin{aligned}
\| \nabla f(x_k) + A_k^\top y_k + \nu_k + w_k \|_\infty \\
\| g(x_k) - s_k \|_\infty \\
\| X_k\nu_k - \mu e \|_\infty \\
\| S_k w_k - \mu e \|_\infty
\end{aligned}
\right\}

```
and stopping the algorithm whenever
```math
E_0(x_k, s_k; y_k, \nu_k, w_k) < \varepsilon_{tol}

```

### User-defined termination criteria

Users can also define a custom termination criteria by using the `intermediate_callback` solver option to provide
a function that returns a boolean value indicating whether to stop.
The function takes two arguments, the `MadNLP` solver and the current mode of the solver, `:regular`, `:restore` or `:robust`.
For example:

```julia
function user_callback_termination(solver::MadNLP.AbstractMadNLPSolver, mode::Symbol)
    # Access solver state: solver.cnt.k, solver.inf_pr, solver.inf_du, etc.
    return solver.cnt.k > 100
end

solver = MadNLPSolver(nlp; intermediate_callback=user_callback_termination)
```

When the callback returns `true`, the solver terminates with status `USER_REQUESTED_STOP`.

## Step 2: How to update the barrier parameter $$\mu$$?
TODO

## Step 4: How do we solve the KKT system?

Solving the KKT system happens in two substeps:
1. Assembling the KKT matrix $$K$$.
2. Solve the system $$Kx = b$$ with a linear solver.

In [substep 1](kkt.md), MadNLP reads the Hessian and the Jacobian
computed at Step 3 and build the associated KKT system
in an `AbstractKKTSystem`. As a result, we get a matrix $$K_k$$
encoding the KKT system at iteration $$k$$.

In [substep 2](linear_solvers.md), the matrix $$K_k$$ is factorized by a compatible
linear solver. Then a solution $$x$$ is returned by applying
a backsolve.

## Step 5: How to regularize the KKT system?
TODO

## Step 6: What is a filter line-search?
TODO


### What happens if the line-search failed?

If inside the line-search algorithm the step $$\alpha_k$$ becomes
negligible ($$<10^{-8}$$) then we consider the line-search
has failed to find a next iterate along the current direction $d_k$.
If that happens during several consecutive iterations, MadNLP
enters into a feasible restoration phase. The goal of feasible
restoration is to decrease the primal infeasibility, to move the
current iterate closer to the feasible set.
