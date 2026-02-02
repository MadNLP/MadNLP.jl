# Installation
To install MadNLP, simply proceed to
```julia
pkg> add MadNLP
```

In addition to MUMPS, SuiteSparse and LAPACK, the user can install the following extensions to use a specialized linear solver.

## HSL linear solver
Obtain a license and download HSL_jll.jl from https://licences.stfc.ac.uk/products/Software/HSL/LibHSL.
Install this download into your current environment using:
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
