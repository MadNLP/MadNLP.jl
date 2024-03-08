![logo](https://github.com/MadNLP/MadNLP.jl/blob/master/logo-full.svg)

*A [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming) solver based on the filter line-search [interior point method](https://en.wikipedia.org/wiki/Interior-point_method) (as in [Ipopt](https://github.com/coin-or/Ipopt)) that can handle/exploit diverse classes of data structures, either on [host](https://en.wikipedia.org/wiki/Central_processing_unit) or [device](https://en.wikipedia.org/wiki/Graphics_processing_unit) memories.*

---

| **License** | **Documentation** | **Build Status** | **Coverage** | **DOI** |
|:-----------------:|:-----------------:|:----------------:|:----------------:|:----------------:|
| [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MadNLP/MadNLP.jl/blob/master/LICENSE) | [![doc](https://img.shields.io/badge/docs-stable-blue.svg)](https://madnlp.github.io/MadNLP.jl/stable) [![doc](https://img.shields.io/badge/docs-dev-blue.svg)](https://madnlp.github.io/MadNLP.jl/dev) | [![build](https://github.com/MadNLP/MadNLP.jl/actions/workflows/test.yml/badge.svg)](https://github.com/MadNLP/MadNLP.jl/actions/workflows/test.yml) | [![codecov](https://codecov.io/gh/MadNLP/MadNLP.jl/branch/master/graph/badge.svg?token=MBxH2AAu8Z)](https://codecov.io/gh/MadNLP/MadNLP.jl) | [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5825776.svg)](https://doi.org/10.5281/zenodo.5825776) |

## Installation

```julia
pkg> add MadNLP
```

Optionally, various extension packages can be installed together:
```julia
pkg> add MadNLPHSL, MadNLPPardiso, MadNLPMumps, MadNLPGPU
```

These packages are stored in the `lib` subdirectory within the main MadNLP repository. Some extension packages may require additional dependencies or specific hardware. For the instructions for the build procedure, see the following links:

 * [MadNLPHSL](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPHSL)
 * [MadNLPMumps](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPMumps)
 * [MadNLPPardiso](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPHSL)
 * [MadNLPGPU](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPGPU)

## Usage

### Interfaces

MadNLP is interfaced with modeling packages:

- [JuMP](https://github.com/jump-dev/JuMP.jl)
- [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

Users can pass various options to MadNLP also through the modeling packages. The interface-specific syntax are shown below. To see the list of MadNLP solver options, check the [documentation](https://madnlp.github.io/MadNLP.jl/dev/options/).

#### JuMP interface

```julia
using MadNLP, JuMP
model = Model(()->MadNLP.Optimizer(print_level=MadNLP.INFO, max_iter=100))
@variable(model, x, start = 0.0)
@variable(model, y, start = 0.0)
@NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)
optimize!(model)
```

#### NLPModels interface

```julia
using MadNLP, CUTEst
model = CUTEstModel("PRIMALC1")
madnlp(model, print_level=MadNLP.WARN, max_wall_time=3600)
```

### Linear Solvers

MadNLP is interfaced with non-Julia sparse/dense linear solvers:
- [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html)
- [Lapack](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/lapack-routines.html)
- [HSL solvers](http://www.hsl.rl.ac.uk/ipopt/) (requires extension)
- [Pardiso](https://www.pardiso-project.org/) (requires extension)
- [Pardiso-MKL](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/sparse-solver-routines/intel-mkl-pardiso-parallel-direct-sparse-solver-interface.html) (requires extension)
- [Mumps](http://mumps.enseeiht.fr/)  (requires extension)
- [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) (requires extension)
- [cuDSS](https://docs.nvidia.com/cuda/cudss/index.html) (requires extension)

Each linear solver in MadNLP is a Julia type, and the `linear_solver` option should be specified by the actual type. Note that the linear solvers are always exported to `Main`.

#### Built-in Solvers: Umfpack, LapackCPU

```julia
using MadNLP, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=UmfpackSolver)) # default
model = Model(()->MadNLP.Optimizer(linear_solver=LDLSolver))     # works only for convex problems
model = Model(()->MadNLP.Optimizer(linear_solver=CHOLMODSolver)) # works only for convex problems
model = Model(()->MadNLP.Optimizer(linear_solver=LapackCPUSolver))
```

#### HSL (requires extension `MadNLPHSL`)

```julia
using MadNLP, MadNLPHSL, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=Ma27Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma57Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma77Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma86Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma97Solver))
```

#### Mumps (requires extension `MadNLPMumps`)

```julia
using MadNLP, MadNLPMumps, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MumpsSolver))
```

#### Pardiso (requires extension `MadNLPPardiso`)

```julia
using MadNLP, MadNLPPardiso, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=PardisoSolver))
model = Model(()->MadNLP.Optimizer(linear_solver=PardisoMKLSolver))
```

#### CUDA (requires extension `MadNLPGPU`)

```julia
using MadNLP, MadNLPGPU, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=LapackGPUSolver))  # for dense problems
model = Model(()->MadNLP.Optimizer(linear_solver=CUDSSSolver))      # for sparse problems
model = Model(()->MadNLP.Optimizer(linear_solver=CuCholeskySolver)) # for sparse problems
model = Model(()->MadNLP.Optimizer(linear_solver=GLUSolver))        # for sparse problems
model = Model(()->MadNLP.Optimizer(linear_solver=RFSolver))         # for sparse problems
```

## Citing MadNLP.jl

If you use MadNLP.jl in your research, we would greatly appreciate your citing it.

```bibtex
@article{shin2023accelerating,
  title={Accelerating optimal power flow with {GPU}s: {SIMD} abstraction of nonlinear programs and condensed-space interior-point methods},
  author={Shin, Sungho and Pacaud, Fran{\c{c}}ois and Anitescu, Mihai},
  journal={arXiv preprint arXiv:2307.16830},
  year={2023}
}
@article{shin2020graph,
  title={Graph-Based Modeling and Decomposition of Energy Infrastructures},
  author={Shin, Sungho and Coffrin, Carleton and Sundar, Kaarthik and Zavala, Victor M},
  journal={arXiv preprint arXiv:2010.02404},
  year={2020}
}
```

## Supporting MadNLP.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/MadNLP/MadNLP.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/MadNLP/MadNLP.jl/discussions).
