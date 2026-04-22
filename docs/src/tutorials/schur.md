# Schur complement KKT system

```@meta
CurrentModule = MadNLP
```
```@setup schur
using LinearAlgebra
using MadNLP
using MadNLPTests
```

This tutorial shows how to use the [`SchurComplementKKTSystem`](@ref) to solve
nonlinear programs with the block-arrowhead structure of two-stage stochastic
programs. The KKT system is reduced per IPM iteration to a dense `nd × nd`
system on the design variables, with each scenario block factorized
independently. The same KKT system runs on the GPU via
[`MadNLPGPU`](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPGPU),
which dispatches the per-scenario factorizations to a batched cuDSS solver and
the Schur accumulation to CUBLAS strided batched GEMM.

## Two-stage stochastic programs

Consider a two-stage stochastic NLP with `ns` scenarios, `nv` recourse
variables and `nc` constraints per scenario, and `nd` first-stage design
variables shared across scenarios:
```math
\begin{aligned}
\min_{v_1,\dots,v_{n_s},\, d}\quad & \sum_{k=1}^{n_s} f_k(v_k, d) + f_0(d) \\
\text{s.t.}\quad & c_k(v_k, d) = 0,\quad k = 1,\dots,n_s \\
& v_k \in [\underline v_k, \overline v_k],\quad d \in [\underline d, \overline d].
\end{aligned}
```
The variable layout `x = [v_1, …, v_{ns}, d]` makes the Hessian and Jacobian
block-arrowhead:
```math
H = \begin{bmatrix}
H_{11} & & & H_{1d} \\
       & \ddots & & \vdots \\
       & & H_{ns\,ns} & H_{ns\,d} \\
H_{1d}^\top & \cdots & H_{ns\,d}^\top & H_{dd}
\end{bmatrix}, \qquad
J = \begin{bmatrix}
J_{11} & & & J_{1d} \\
& \ddots & & \vdots \\
& & J_{ns\,ns} & J_{ns\,d}
\end{bmatrix}.
```
Stacking each scenario's primal block with its equality dual block gives an
augmented per-scenario block ``A_k`` of size ``\text{blk} = n_v + n_c^{eq}``,
coupled to the design variables only through the dense block ``C_{dk}``.
[`SchurComplementKKTSystem`](@ref) factors each ``A_k`` once per IPM iteration
and accumulates the dense Schur complement
```math
S = H_{dd} + \Sigma_d - \sum_{k=1}^{n_s} C_{dk} A_k^{-1} C_{dk}^\top
```
where ``\Sigma_d`` collects the bound-multiplier contributions on the design
variables. ``S`` is dense, ``n_d \times n_d``, and is factored by a dense
linear solver (LAPACK on CPU, cuSOLVER on GPU). Inequality constraints are
condensed into the per-scenario blocks before factorization.

## Building a two-stage QP

For a self-contained example we use `MadNLPTests.build_twostage_qp`, which
constructs a diagonal-Hessian two-stage QP with the variable and constraint
ordering expected by the Schur system. Consider the toy stochastic LP:
```math
\min_{v, d}\; \sum_{k=1}^{n_s} (v_k - \theta_k)^2 + (d - 1)^2
\quad\text{s.t.}\quad v_k + d = 0,\; k = 1,\dots,n_s.
```

```@example schur
ns, nv, nd, nc = 3, 1, 1, 1
θ = [4.0, 6.0, 8.0]

qp = build_twostage_qp(;
    ns, nv, nd, nc,
    hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
    g_v    = reshape(-2 .* θ, nv, ns), g_d = [-2.0],
    A_v    = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
    lcon   = zeros(nc, ns), ucon = zeros(nc, ns),
    lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
    lvar_d = fill(-100.0, nd),     uvar_d = fill(100.0, nd),
)
nothing
```

## Solving with `SchurComplementKKTSystem`

[`SchurComplementKKTSystem`](@ref) needs to know the four scenario dimensions
`(ns, nv, nd, nc)` so it can carve the global sparsity into per-scenario
blocks. They are passed through the `kkt_options` argument of `madnlp`. The
helper [`MadNLPTests.schur_opts`](@ref) just packs them into the expected
dictionary.

```@example schur
results = madnlp(
    qp;
    kkt_system    = SchurComplementKKTSystem,
    linear_solver = LapackCPUSolver,
    kkt_options   = schur_opts(; ns, nv, nd, nc),
    print_level   = MadNLP.ERROR,
)
results.solution
```

The analytic optimum is ``d^\star = (1 - \sum_k \theta_k) / (n_s + 1)`` and
``v_k^\star = -d^\star``:

```@example schur
d_star = (1.0 - sum(θ)) / (ns + 1)
@assert isapprox(results.solution[end], d_star; atol = 1.0e-3)
@assert all(isapprox(results.solution[k], -d_star; atol = 1.0e-3) for k in 1:ns)
nothing
```

!!! info
    The Schur system reduces the linear-algebra cost from one factorization of
    a single sparse system of size ``n_s(n_v + n_c) + n_d`` to ``n_s``
    independent factorizations of size ``n_v + n_c^{eq}`` plus one dense
    factorization of size ``n_d``. The per-scenario factorizations are
    independent, which is what makes the GPU variant a batched call. The
    payoff grows with `ns` and shrinks with `nd`.

### Choosing the per-scenario solver

By default each per-scenario block is factored by [`LDLSolver`](@ref). On the
CPU you can swap in any sparse solver (e.g. an HSL solver) via the
`schur_scenario_linear_solver` option; the dense Schur factor is controlled by
the usual `linear_solver` argument:

```julia
results = madnlp(
    qp;
    kkt_system    = SchurComplementKKTSystem,
    linear_solver = LapackCPUSolver,
    kkt_options   = Dict{Symbol, Any}(
        :schur_ns => ns, :schur_nv => nv, :schur_nd => nd, :schur_nc => nc,
        :schur_scenario_linear_solver => MadNLP.LDLSolver,
    ),
)
```

## Running on the GPU

`SchurComplementKKTSystem` is backend-agnostic: building the model with a GPU
storage type and selecting the GPU dense solver is enough. Per-scenario
factorizations dispatch to a batched cuDSS solver and the Schur complement is
accumulated through a strided batched GEMM. The model still has to be built
through the `SparseCallback` since the per-scenario blocks are sparse.

```julia
using CUDA, CUDSS, MadNLPGPU

nlp = build_twostage_qp(
    CUDA.zeros(Float64, ns * nv + nd);
    ns, nv, nd, nc,
    hess_v = fill(2.0, nv, ns), hess_d = fill(2.0, nd),
    g_v    = reshape(-2 .* θ, nv, ns), g_d = [-2.0],
    A_v    = fill(1.0, nc, nv, ns), A_d = fill(1.0, nc, nd, ns),
    lcon   = zeros(nc, ns), ucon = zeros(nc, ns),
    lvar_v = fill(-100.0, nv, ns), uvar_v = fill(100.0, nv, ns),
    lvar_d = fill(-100.0, nd),     uvar_d = fill(100.0, nd),
)

results = madnlp(
    nlp;
    callback      = MadNLP.SparseCallback,
    kkt_system    = SchurComplementKKTSystem,
    linear_solver = LapackCUDASolver,
    kkt_options   = schur_opts(; ns, nv, nd, nc),
)
```

## Auto-detecting dimensions from ExaModels

When the model is built with an ExaModels `TwoStageExaCore`, the variable and
constraint scenario tags are attached to the model and
[`SchurComplementKKTSystem`](@ref) can recover `(ns, nv, nd, nc)` directly from
them. The convention is `tags.var_scenario[i] == 0` for design variables,
`tags.var_scenario[i] == k` (with `k ∈ 1:ns`) for scenario `k` recourse
variables, and the same encoding for `tags.con_scenario`. When the tags are
present, you can omit `kkt_options` entirely:

```julia
using ExaModels

core = ExaModels.TwoStageExaCore(Float64)
d  = ExaModels.variable(core, nd; lvar = -100.0, uvar = 100.0, scenario = 0)
for k in 1:ns
    v = ExaModels.variable(core, nv; lvar = -100.0, uvar = 100.0, scenario = k)
    ExaModels.objective(core, (v[j] - θ[k])^2 for j in 1:nv)
    ExaModels.constraint(core, v[j] + d[j] for j in 1:nv; scenario = k)
end
ExaModels.objective(core, (d[j] - 1.0)^2 for j in 1:nd)
nlp = ExaModels.ExaModel(core)

results = madnlp(nlp; kkt_system = SchurComplementKKTSystem)
```

## Variable and constraint ordering

The Schur reduction relies on a strict block layout — `madnlp` will throw an
assertion error if the dimensions don't match. The expected ordering is

- variables: `[v_{1,1}, …, v_{1,nv}, v_{2,1}, …, v_{ns,nv}, d_1, …, d_nd]`,
  i.e. all scenarios first (contiguous per scenario), design variables last;
- constraints: `[c_{1,1}, …, c_{1,nc}, …, c_{ns,1}, …, c_{ns,nc}]`, contiguous
  per scenario.

When using ExaModels with `TwoStageExaCore`, declaring variables and
constraints with the right `scenario` tag is enough — the tags also drive the
auto-detection above.
