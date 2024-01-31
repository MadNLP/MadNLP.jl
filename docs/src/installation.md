# Installation
To install MadNLP, simply proceed to
```julia
pkg> add MadNLP

```

!!! note
    The default installation comes is shipped only with two linear solvers
    (Umfpack and Lapack), which are not adapted to solve the KKT systems
    arising in large-scale nonlinear problems.
    We recommend using a specialized linear solver to speed-up the solution of
    the KKT systems.

In addition to Lapack and Umfpack, the user can install the following extensions to
use a specialized linear solver.

## HSL linear solver
Obtain a license and download HSL_jll.jl from https://licences.stfc.ac.uk/product/julia-hsl. There are two versions available: LBT and OpenBLAS. LBT is the recommended option for Julia >= v1.9. Install this download into your current environment using:
```julia
import Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
```

If the user has already compiled the HSL solver library, one can
simply override the path to the artifact by editing `~/.julia/artifacts/Overrides.toml`
```
# replace HSL_jll artifact /usr/local/lib/libhsl.so
ecece3e2c69a413a0e935cf52e03a3ad5492e137 = "/usr/local"
```

## Mumps linear solver

Mumps is an open-source sparse linear solver, whose binaries are kindly
provided as a Julia artifact.
Installing Mumps simply amounts to
```julia
pkg> add MadNLPMumps
```

## Pardiso linear solver

To use Pardiso, the user needs to obtain the Pardiso shared libraries from
<https://panua.ch/>, provide the absolute path to the shared library:
```
julia> ENV["MADNLP_PARDISO_LIBRARY_PATH"] = "/usr/lib/libpardiso600-GNU800-X86-64.so"
```
and place the license file in the home directory.
After obtaining the library and the license file, run
```julia
pkg> build MadNLPPardiso
```

The build process requires a C compiler.

