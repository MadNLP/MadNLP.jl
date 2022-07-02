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
If the user has access to HSL, we recommend using this set of linear
solver inside the interior-point algorithm.

To build MadNLP with HSL linear solvers (Ma27, Ma57, Ma77, Ma86, Ma97), the
source codes need to be obtained by the user from
<http://www.hsl.rl.ac.uk/ipopt/> under Coin-HSL Full (Stable). The source
codes are distributed as a tarball file `coinhsl-*.tar.gz`. Once
uncompressed, the absolute path to the extracted source code should be specified as:
```julia
julia> ENV["MADNLP_HSL_SOURCE_PATH"] = "/opt/coinhsl"
```

If the user has already compiled the HSL solver library, one can
simply provide a path to the compiled shared library (in this case, the source code is
not compiled and the provided shared library is directly used):
```julia
julia> ENV["MADNLP_HSL_LIBRARY_PATH"] = "/usr/lib/libcoinhsl.so"
```

Once the environment variable set, build `MadNLPHSL` with
```julia
pkg> build MadNLPHSL
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
<https://www.pardiso-project.org/>, provide the absolute path to the shared library:
```
julia> ENV["MADNLP_PARDISO_LIBRARY_PATH"] = "/usr/lib/libpardiso600-GNU800-X86-64.so"
```
and place the license file in the home directory.
After obtaining the library and the license file, run
```julia
pkg> build MadNLPPardiso
```

The build process requires a C compiler.

