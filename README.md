![logo](https://github.com/MadNLP/MadNLP.jl/blob/master/logo-full.svg)

*A [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming) solver based on the filter line-search [interior point method](https://en.wikipedia.org/wiki/Interior-point_method) (as in [Ipopt](https://github.com/coin-or/Ipopt)) that can handle/exploit diverse classes of data structures, either on [host](https://en.wikipedia.org/wiki/Central_processing_unit) or [device](https://en.wikipedia.org/wiki/Graphics_processing_unit) memories.*

---

| **License** | **Documentation** | **Build Status** | **Coverage** | **DOI** |
|:-----------:|:-----------------:|:----------------:|:------------:|:-------:|
| [![License: MIT][license-img]][license-url] | [![docs-stable][docs-stable-img]][docs-stable-url] [![docs-dev][docs-dev-img]][docs-dev-url] | [![build-gh][build-gh-img]][build-gh-url] | [![codecov][codecov-img]][codecov-url] | [![doi][doi-img]][doi-url] |

[license-img]: https://img.shields.io/badge/License-MIT-yellow.svg
[license-url]: https://github.com/MadNLP/MadNLP.jl/blob/master/LICENSE
[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://madnlp.github.io/MadNLP.jl/stable
[docs-dev-img]: https://img.shields.io/badge/docs-dev-purple.svg
[docs-dev-url]: https://madnlp.github.io/MadNLP.jl/dev
[build-gh-img]: https://github.com/MadNLP/MadNLP.jl/actions/workflows/test.yml/badge.svg
[build-gh-url]: https://github.com/MadNLP/MadNLP.jl/actions/workflows/test.yml
[codecov-img]: https://codecov.io/gh/MadNLP/MadNLP.jl/branch/master/graph/badge.svg?token=MBxH2AAu8Z
[codecov-url]: https://codecov.io/gh/MadNLP/MadNLP.jl
[doi-img]: https://zenodo.org/badge/DOI/10.5281/zenodo.5825776.svg
[doi-url]: https://doi.org/10.5281/zenodo.5825776

## Installation

```julia
pkg> add MadNLP
```

Optionally, various extension packages can be installed together:
```julia
pkg> add MadNLPHSL, MadNLPPardiso, MadNLPGPU
```

These packages are stored in the `lib` subdirectory within the main MadNLP repository.
Some extension packages may require additional dependencies or specific hardware.
For the instructions for the build procedure, see the following links:

 * [MadNLPHSL](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPHSL)
 * [MadNLPPardiso](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPHSL)
 * [MadNLPGPU](https://github.com/MadNLP/MadNLP.jl/tree/master/lib/MadNLPGPU)

## Usage

### Interfaces

MadNLP is interfaced with modeling packages:

- [JuMP](https://github.com/jump-dev/JuMP.jl)
- [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

Users can pass various options to MadNLP also through the modeling packages.
The interface-specific syntax are shown below.
To see the list of MadNLP solver options, check the [documentation](https://madnlp.github.io/MadNLP.jl/dev/options/).

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

MadNLP is interfaced with non-Julia direct sparse/dense linear solvers:
- [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html)
- [Lapack](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/lapack-routines.html)
- [HSL solvers](http://www.hsl.rl.ac.uk/ipopt/) (requires extension `MadNLPHSL`)
- [Pardiso](https://www.pardiso-project.org/) (requires extension `MadNLPPardiso`)
- [Pardiso-MKL](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/sparse-solver-routines/intel-mkl-pardiso-parallel-direct-sparse-solver-interface.html) (requires extension `MadNLPPardiso`)
- [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) (requires extension `MadNLPGPU`)
- [cuDSS](https://docs.nvidia.com/cuda/cudss/index.html) (requires extension `MadNLPGPU`)
- [rocSOLVER](https://rocm.docs.amd.com/projects/rocSOLVER/en/latest/) (requires extension `MadNLPGPU`)

Each linear solver in MadNLP is a Julia type, and the `linear_solver` option should be specified by the actual type. Note that the linear solvers are always exported to `Main`.

#### Built-in Solvers: Umfpack, LapackCPU

```julia
using MadNLP, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MumpsSolver))   # default
model = Model(()->MadNLP.Optimizer(linear_solver=UmfpackSolver))
model = Model(()->MadNLP.Optimizer(linear_solver=LDLSolver))     # works only for convex problems
model = Model(()->MadNLP.Optimizer(linear_solver=CHOLMODSolver)) # works only for convex problems
model = Model(()->MadNLP.Optimizer(linear_solver=LapackCPUSolver))
```

#### HSL (requires extension `MadNLPHSL`)

```julia
using MadNLPHSL, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=Ma27Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma57Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma77Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma86Solver))
model = Model(()->MadNLP.Optimizer(linear_solver=Ma97Solver))
```

#### Pardiso (requires extension `MadNLPPardiso`)

```julia
using MadNLPPardiso, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=PardisoSolver))
model = Model(()->MadNLP.Optimizer(linear_solver=PardisoMKLSolver))
```

#### CUDA and ROCm (requires extension `MadNLPGPU`)

```julia
using MadNLPGPU, JuMP

using CUDA
model = Model(()->MadNLP.Optimizer(linear_solver=LapackCUDASolver, array_type=CuArray))   # for dense problems
model = Model(()->MadNLP.Optimizer(linear_solver=CUDSSSolver, array_type=CuArray))        # for sparse problems

using AMDGPU
model = Model(()->MadNLP.Optimizer(linear_solver=LapackROCmSolver, array_type=ROCArray))  # for dense problems
```

## Citing MadNLP.jl

If you use MadNLP.jl in your research, we would greatly appreciate your citing it.

```bibtex
@article{shin2024accelerating,
  title     = {Accelerating optimal power flow with {GPU}s: {SIMD} abstraction of nonlinear programs and condensed-space interior-point methods},
  author    = {Shin, Sungho and Anitescu, Mihai and Pacaud, Fran{\c{c}}ois},
  journal   = {Electric Power Systems Research},
  volume    = {236},
  pages     = {110651},
  year      = {2024},
  publisher = {Elsevier}
}

@article{shin2021graph,
  title     = {Graph-based modeling and decomposition of energy infrastructures},
  author    = {Shin, Sungho and Coffrin, Carleton and Sundar, Kaarthik and Zavala, Victor M},
  journal   = {IFAC-PapersOnLine},
  volume    = {54},
  number    = {3},
  pages     = {693--698},
  year      = {2021},
  publisher = {Elsevier}
}
```

## Supporting MadNLP.jl
- Please report issues and feature requests via the [GitHub issue tracker](https://github.com/MadNLP/MadNLP.jl/issues).
- Questions are welcome at [GitHub discussion forum](https://github.com/MadNLP/MadNLP.jl/discussions).
