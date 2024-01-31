![Logo](logo-full.svg)

MadNLP is an open-source nonlinear programming solver,
purely implemented in Julia. MadNLP implements a filter line-search
interior-point algorithm, as used in [Ipopt](https://github.com/coin-or/Ipopt). MadNLP
seeks to streamline the development of modeling and algorithmic paradigms in
order to exploit structures and to make efficient use of high-performance computers.


## Design

### MadNLP's problem structure
MadNLP targets the resolution of constrained nonlinear problems,
formulating as
```math
  \begin{aligned}
    \min_{x} \; & f(x) \\
    \text{subject to} \quad & g_\ell \leq g(x) \leq g_u \\
                            & x_\ell \leq x \leq x_u
  \end{aligned}
```
where $$x \in \mathbb{R}^n$$ is the decision variable, $$f: \mathbb{R}^n \to \mathbb{R}$$
and $$g: \mathbb{R}^n \to \mathbb{R}^m$$ two nonlinear functions.
MadNLP makes the distinction between the **bound constraints** $$x_\ell \leq x \leq x_u$$
and the **generic constraints** $$g_\ell \leq g(x) \leq g_u$$.
No other structure is assumed _a priori_.

MadNLP is built on top of [NLPModels.jl](https://github.com/JuliaSmoothOptimizers/NLPModels.jl/),
a generic package to represent optimization models in Julia. In addition,
MadNLP is interfaced with [JuMP](https://github.com/jump-dev/JuMP.jl) and
[Plasmo](https://github.com/zavalab/Plasmo.jl).

!!! note
    MadNLP requires the evaluation of both the first-order and the second-order
    derivatives of the nonlinear problem.


### MadNLP's algorithm

MadNLP implements a primal-dual filter line-search interior-point algorithm,
closely related to [Ipopt](https://github.com/coin-or/Ipopt).
The nonlinear problem is reformulated in standard form by introducing
a slack variables $$s \in \mathbb{R}^m$$ to rewrite all the inequality
constraints as equality constraints:
```math
  \begin{aligned}
    \min_{x, s} \; & f(x) \\
    \text{subject to} \quad & g(x) - s = 0  \\
                            & g_\ell \leq s \leq g_u \\
                            & x_\ell \leq x \leq x_u
  \end{aligned}
```

The algorithm starts from an initial primal-dual iterate $$(x_0, s_0, y_0)$$.
Then, at each iteration the current iterate is updated by solving the
KKT linear system:
```math
\begin{bmatrix}
    W_k + \Sigma_x & 0 & A_k^\top \\
    0 & \Sigma_s & - I \\
    A_k & -I & 0
\end{bmatrix}
\begin{bmatrix}
    \Delta x \\ \Delta s \\ \Delta y
\end{bmatrix}
=
-
\begin{bmatrix}
    \nabla f(x_k) + A_k^\top y_k - \mu U^{-1} e_n \\
    y_k - \mu S^{-1} e_n \\
    g(x_k) - s_k
\end{bmatrix}

```
with $$\mu$$ being the current barrier parameter.

We call the linear system the **augmented KKT system** at iteration $$k$$.


### MadNLP's performance
In any interior-point algorithm, the two computational bottlenecks are
1. The evaluation of the first- and second-order derivatives.
2. The factorization of the augmented KKT system.

The first point is problem dependent, and often related to the
automatic differentiation backend being employed.
The second point is more problematic, as the augmented KKT system
is usually symmetric indefinite and ill-conditionned.
For that reason we recommend using efficient sparse-linear solvers
(HSL, Mumps, Pardiso) if they are available to the user.



## Citing MadNLP.jl
If you use MadNLP.jl in your research, we would greatly appreciate your citing it.

```bibtex
@article{shin2020graph,
  title={Graph-Based Modeling and Decomposition of Energy Infrastructures},
  author={Shin, Sungho and Coffrin, Carleton and Sundar, Kaarthik and Zavala, Victor M},
  journal={arXiv preprint arXiv:2010.02404},
  year={2020}
}
```
