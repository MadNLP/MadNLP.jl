MadNLP
========
[![Build Status](https://travis-ci.org/sshin23/MadNLP.jl.svg?branch=master)](https://travis-ci.org/sshin23/MadNLP.jl) [![codecov](https://codecov.io/gh/sshin23/MadNLP.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/sshin23/MadNLP.jl)


MadNLP is a [nonlinear programming](https://en.wikipedia.org/wiki/Nonlinear_programming) (NLP) solver, purely implemented in [Julia](https://julialang.org/). MadNLP implements a filter line-search algorithm, as that used in [Ipopt](https://github.com/coin-or/Ipopt). MadNLP seeks to streamline the development of modeling and algorithmic paradigms in order to exploit structures and to make efficient use of high-performance computers. 

## Installation
```julia
pkg> add git@github.com:sshin23/MadNLP.git
```

## Build
**Automatic build is currently only supported for Linux and MacOS.**

The build process requires C and Fortran compilers. If they are not installed, do
```julia
shell> sudo apt install gcc # Linux
shell> brew install gcc # MacOS
```

MadNLP is interfaced with non-Julia sparse/dense linear solvers:
- [Umfpack](https://people.engr.tamu.edu/davis/suitesparse.html)
- [Mumps](http://mumps.enseeiht.fr/) 
- [MKL-Pardiso](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/sparse-solver-routines/intel-mkl-pardiso-parallel-direct-sparse-solver-interface.html) 
- [HSL solvers](http://www.hsl.rl.ac.uk/ipopt/) (optional) 
- [Pardiso](https://www.pardiso-project.org/) (optional) 
- [MKL-Lapack](https://software.intel.com/content/www/us/en/develop/documentation/mkl-developer-reference-fortran/top/lapack-routines.html)
- [cuSOLVER](https://docs.nvidia.com/cuda/cusolver/index.html) (optional)

All the dependencies except for HSL solvers, Pardiso, and CUDA are automatically installed. To build MadNLP with HSL linear solvers (Ma27, Ma57, Ma77, Ma86, Ma97), the source codes need to be obtained by the user from <http://www.hsl.rl.ac.uk/ipopt/> under Coin-HSL Full (Stable). Then, the tarball `coinhsl-2015.06.23.tar.gz` should be placed at `deps/download`. To use Pardiso, the user needs to obtain the Paridso shared libraries from <https://www.pardiso-project.org/>, place the shared library file (e.g., `libpardiso600-GNU720-X86-64.so`) at `deps/download`, and place the license file in the home directory. To use cuSOLVER, functional NVIDIA driver and corresponding CUDA toolkit need to be installed by the user. After obtaining the files, run
```julia
pkg> build MadNLP
```
Build can be customized by setting the following environment variables.
```julia
julia> ENV["MADNLP_CC"] = "/usr/local/bin/gcc-9"    # C compiler
julia> ENV["MADNLP_FC"] = "/usr/local/bin/gfortran" # Fortran compiler
julia> ENV["MADNLP_BLAS"] = "openblas"              # default is MKL
julia> ENV["MADNLP_ENALBE_OPENMP"] = false          # default is true
julia> ENV["MADNLP_OPTIMIZATION_FLAG"] = "-O2"      # default is -O3
```

## Usage
MadNLP is interfaced with modeling packages: 
- [JuMP](https://github.com/jump-dev/JuMP.jl)
- [Plasmo](https://github.com/zavalab/Plasmo.jl)
- [NLPModels](https://github.com/JuliaSmoothOptimizers/NLPModels.jl).

### JuMP interface
```julia
using MadNLP, JuMP

model = Model(()->MadNLP.Optimizer(linear_solver="ma57",log_level="info",max_iter=100))
@variable(model, x, start = 0.0)
@variable(model, y, start = 0.0)
@NLobjective(model, Min, (1 - x)^2 + 100 * (y - x^2)^2)

optimize!(model)

```

### Plasmo interface
```julia
using MadNLP, Plasmo

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

MadNLP.optimize!(graph,ipopt;linear_solver="ma97",log_level="debug",max_iter=100)

```

### NLPModels interface
```julia
using MadNLP, CUTEst
model = CUTEstModel("PRIMALC1")
plamonlp(model,linear_solver="pardisomkl",log_level="warn",max_wall_time=3600)
```

## Solver options
To see the list of MadNLP solver options, check the [OPTIONS.md](https://github.com/sshin23/MadNLP/blob/master/OPTIONS.md) file.
## Bug reports and support
Please report issues and feature requests via the [Github issue tracker](https://github.com/sshin23/MadNLP/issues).
