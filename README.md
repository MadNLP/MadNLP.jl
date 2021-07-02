![Logo](logo-full.svg) 
---

[![build](https://github.com/sshin23/MadNLP.jl/workflows/build/badge.svg?branch=dev%2Fgithub_actions)](https://github.com/sshin23/MadNLP.jl/actions?query=workflow%3Abuild) [![codecov](https://codecov.io/gh/sshin23/MadNLP.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sshin23/MadNLP.jl)

MadNLP is a [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLP) solver, purely implemented in [Julia](https://julialang.org/). MadNLP implements a filter line-search algorithm, as that used in [Ipopt](https://github.com/coin-or/Ipopt). MadNLP seeks to streamline the development of modeling and algorithmic paradigms in order to exploit structures and to make efficient use of high-performance computers. 

## Installation
```julia
pkg> add MadNLP
```
Optionally, various extension packages can be installed together:
```julia
pkg> add MadNLPHSL, MadNLPPardiso, MadNLPMumps, MadNLPGPU, MadNLPGraphs, MadNLPIterative
```
These packages are stored in the `lib` subdirectory within the main MadNLP repository. Some extension packages may require additional dependencies or specific hardware. For the instructions for the build procedure, see the following links: [MadNLPHSL](https://github.com/sshin23/MadNLP.jl/tree/master/lib/MadNLPHSL), [MadNLPPardiso](https://github.com/sshin23/MadNLP.jl/tree/master/lib/MadNLPHSL), [MadNLPGPU](https://github.com/sshin23/MadNLP.jl/tree/master/lib/MadNLPGPU).


## Usage
### Interfaces
MadNLP is interfaced with modeling packages: 
- [JuMP](https://github.com/jump-dev/JuMP.jl)
- [Plasmo](https://github.com/zavalab/Plasmo.jl)
- [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).
Users can pass various options to MadNLP also through the modeling packages. The interface-specific syntaxes are shown below. To see the list of MadNLP solver options, check the [OPTIONS.md](https://github.com/sshin23/MadNLP/blob/master/OPTIONS.md) file.

#### JuMP interface
```julia
using MadNLP, JuMP

model = Model(()->MadNLP.Optimizer(print_level=MadNLP.INFO,max_iter=100))
@variable(model, x, start = 0.0)
@variable(model, y, start = 0.0)
@NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

optimize!(model)

```

#### NLPModels interface
```julia
using MadNLP, CUTEst
model = CUTEstModel("PRIMALC1")
madnlp(model,print_level=MadNLP.WARN,max_wall_time=3600)
```

#### Plasmo interface (requires extension `MadNLPGraphs`)
```julia
using MadNLP, MadNLPGraphs, Plasmo

graph = OptiGraph()
@optinode(graph,n1)
@optinode(graph,n2)
@variable(n1,0 <= x <= 2)
@variable(n1,0 <= y <= 3)
@constraint(n1,x+y <= 4)
@objective(n1,Min,x)
@variable(n2,x)
@NLnodeconstraint(n2,exp(x) >= 2)
@linkconstraint(graph,n1[:x] == n2[:x])

MadNLP.optimize!(graph;print_level=MadNLP.DEBUG,max_iter=100)

```

### Linear Solvers
MadNLP is interfaced with non-Julia sparse/dense linear solvers:
- [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html)
- [MKL-Pardiso](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/sparse-solver-routines/intel-mkl-pardiso-parallel-direct-sparse-solver-interface.html)
- [MKL-Lapack](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/lapack-routines.html)
- [HSL solvers](http://www.hsl.rl.ac.uk/ipopt/) (requires extension)
- [Pardiso](https://www.pardiso-project.org/) (requires extension)
- [Mumps](http://mumps.enseeiht.fr/)  (requires extension)
- [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) (requires extension)

Each linear solver in MadNLP is a julia module, and the `linear_solver` option should be specified by the actual module. Note that the linear solver modules are always exported to `Main`.

#### Built-in Solvers: Umfpack, PardisoMKL, LapackCPU
```julia
using MadNLP, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPUmfpack)) # default
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPPardisoMKL))
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPLapackCPU))
```

#### HSL (requires extension `MadNLPHSL`)
```julia
using MadNLP, MadNLPHSL, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPMa27))
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPMa57))
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPMa77))
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPMa86))
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPMa97))
```

#### Mumps (requires extension `MadNLPMumps`)
```julia
using MadNLP, MadNLPMumps, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPMumps))
```

#### Pardiso (requires extension `MadNLPPardiso`)
```julia
using MadNLP, MadNLPPardiso, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPPardiso))
```

#### LapackGPU (requires extension `MadNLPGPU`)
```julia
using MadNLP, MadNLPGPU, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPLapackGPU))
```


#### Schur and Schwarz (requires extension `MadNLPGraphs`)
```julia
using MadNLP, MadNLPGraphs, JuMP
# ...
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPSchwarz))
model = Model(()->MadNLP.Optimizer(linear_solver=MadNLPSchur))
```
The solvers in `MadNLPGraphs` (`Schur` and `Schwawrz`) use multi-thread parallelism; thus, julia session should be started with `-t` flag.
```sh
julia -t 16 # to use 16 threads
```

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

## Bug reports and support
Please report issues and feature requests via the [Github issue tracker](https://github.com/sshin23/MadNLP/issues).
